"""
app.py
------
FastAPI application — the main chatbot API entry point.

Endpoints:
  POST /chat          → AI answer using Gemini + ChromaDB (original)
  POST /chat/hybrid   → AI answer using Qwen2.5:7b + BM25 + BGE-M3 (new)
  GET  /health        → Health check (also verifies vector store is loaded)
  GET  /health/hybrid → Health check for the new hybrid pipeline

Usage:
  cd <project root>
  uvicorn backend.app:app --reload --port 8000
"""


import os
import sys
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Original pipeline (Gemini + ChromaDB) ─────────────────────────────────────
from backend.rag.retrieval.retriever import retrieve
from backend.rag.generation.generator import generate_answer

# ── New hybrid pipeline (Qwen2.5:7b + BM25 + BGE-M3) ─────────────────────────
from bm25_rag.hybrid_bge_qwen_retriever  import hybrid_retrieve
from bm25_rag.hybrid_bge_qwen_generator  import qwen_generate



# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Istacherni — Algerian Law Chatbot API",
    description="RAG-powered chatbot for Algerian law. Ask questions in Arabic.",
    version="1.0.0",
)

# Allow requests from any origin (update in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="ما هي عقوبة السرقة في القانون الجزائري؟",
        description="The legal question to answer (in Arabic or French)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of law articles to retrieve (1–20)",
    )


class ArticleSource(BaseModel):
    id: str
    law_name: str
    law_domain: str
    article_number: str
    title: str
    text_original: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ArticleSource]


# ── Hybrid pipeline request/response models ────────────────────────────────────
class HybridChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="ما هي عقوبة السرقة في القانون الجزائري؟",
        description="The legal question (Arabic / French / English)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of law articles to retrieve (1–20)",
    )


class HybridArticleSource(BaseModel):
    id: str
    law_name: str
    law_domain: str
    article_number: str
    title: str
    text_original: str
    score: float          # RRF fused score


class HybridChatResponse(BaseModel):
    question: str
    answer: str
    model: str            # which LLM answered
    retriever: str        # which retrieval method was used
    sources: list[HybridArticleSource]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    """Verify the server is running and the vector store is accessible."""
    try:
        import chromadb
        from backend.rag.embedding.embed_articles import VECTORSTORE_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        col = client.get_collection(COLLECTION_NAME)
        count = col.count()
        return {
            "status": "ok",
            "vector_store": str(VECTORSTORE_DIR),
            "collection": COLLECTION_NAME,
            "indexed_articles": count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not ready: {str(e)}. Run the ingestion pipeline first.",
        )


@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
def chat(request: ChatRequest):
    """
    Answer a legal question using the RAG pipeline.

    1. Retrieves the top-K most relevant law articles from ChromaDB
    2. Sends them with the question to Gemini for answer generation
    3. Returns the answer + source articles
    """
    try:
        # Retrieve relevant articles
        retrieved_chunks = retrieve(request.question, top_k=request.top_k)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant articles found. Make sure the ingestion pipeline has been run.",
            )

        # Generate answer
        answer = generate_answer(request.question, retrieved_chunks)

        # Build source list
        sources = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            sources.append(ArticleSource(
                id=chunk.get("id", ""),
                law_name=meta.get("law_name", ""),
                law_domain=meta.get("law_domain", ""),
                article_number=meta.get("article_number", ""),
                title=meta.get("title", ""),
                text_original=meta.get("text_original", ""),
                score=chunk.get("score", 0.0),
            ))

        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=sources,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}",
        )


# ── Hybrid pipeline endpoint ──────────────────────────────────────────────────
@app.post("/chat/hybrid", response_model=HybridChatResponse, tags=["hybrid-chatbot"])
def chat_hybrid(request: HybridChatRequest):
    """
    Answer a legal question using the Hybrid RAG pipeline.

    Pipeline:
      1. Retrieve relevant articles via BM25 + BAAI/BGE-M3 (RRF fusion)
      2. Generate the answer locally with Qwen2.5:7b (via Ollama)
      3. Return answer + source articles

    Requirements:
      - Ollama must be running: `ollama serve`
      - Model must be pulled : `ollama pull qwen2.5:7b`
      - BGE-M3 vector store  : run `python rag_bge/bge_embed_articles.py` first
      - BM25 index           : run `python bm25_rag/bm25_indexer.py` first
    """
    try:
        # ── Step 1: Hybrid retrieval (BM25 + BGE-M3 with RRF) ────────────────
        retrieved_chunks = hybrid_retrieve(request.question, top_k=request.top_k)



        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="لم أجد أي مواد قانونية متعلقة بسؤالك في قاعدة البيانات."
            )



        # ── Step 2: Generate answer with Qwen2.5:7b ───────────────────────────
        answer = qwen_generate(request.question, retrieved_chunks)

        # ── Step 3: Build source list ─────────────────────────────────────────
        sources = [
            HybridArticleSource(
                id=chunk.get("id", ""),
                law_name=chunk.get("law_name", ""),
                law_domain=chunk.get("law_domain", ""),
                article_number=chunk.get("article_number", ""),
                title=chunk.get("title", ""),
                text_original=chunk.get("text_original", ""),
                score=chunk.get("score", 0.0),
            )
            for chunk in retrieved_chunks
        ]

        return HybridChatResponse(
            question=request.question,
            answer=answer,
            model="qwen2.5:7b (Ollama)",
            retriever="BM25 + BAAI/BGE-M3 (RRF)",
            sources=sources,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        # Ollama unreachable or model not pulled
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health/hybrid", tags=["system"])
def health_hybrid():
    """Check that the BM25 index, BGE-M3 store, and Ollama are all reachable."""
    import urllib.request, urllib.error, json as _json

    issues = []

    # 1. Check BM25 index files
    from bm25_rag.bm25_indexer import BM25_FILE, CORPUS_FILE
    if not BM25_FILE.exists():
        issues.append(f"BM25 index not found: {BM25_FILE}")
    if not CORPUS_FILE.exists():
        issues.append(f"BM25 corpus not found: {CORPUS_FILE}")

    # 2. Check BGE-M3 vector store
    from bm25_rag.hybrid_bge_qwen_retriever import BGE_VECTORSTORE_DIR, BGE_COLLECTION_NAME
    if not BGE_VECTORSTORE_DIR.exists():
        issues.append(f"BGE-M3 vector store not found: {BGE_VECTORSTORE_DIR}")

    # 3. Ping Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_ok = False
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as r:
            ollama_ok = r.status == 200
    except Exception as e:
        issues.append(f"Ollama unreachable at {ollama_url}: {e}")

    if issues:
        raise HTTPException(status_code=503, detail={"issues": issues})

    return {
        "status": "ok",
        "bm25_index": str(BM25_FILE),
        "bge_vectorstore": str(BGE_VECTORSTORE_DIR),
        "ollama": ollama_url,
        "model": ollama_model,
    }


# ── Entry point (for direct `python backend/app.py` execution) ─────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
