"""
app.py
------
FastAPI application — the main chatbot API entry point.

Endpoints:
  POST /chat       → Return an AI-generated legal answer + source articles
  GET  /health     → Health check (also verifies vector store is loaded)

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

from backend.rag.retrieval.retriever import retrieve
from backend.rag.generation.generator import generate_answer

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


# ── Entry point (for direct `python backend/app.py` execution) ─────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
