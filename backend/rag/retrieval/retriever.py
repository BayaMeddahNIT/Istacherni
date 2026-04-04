"""
retriever.py
------------
Retrieval module: given a user query, embed it and find the top-K most
relevant law article chunks from ChromaDB.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from google import genai
from google.genai import types as genai_types
import chromadb

from backend.rag.preprocessing.normalize_arabic import normalize_arabic

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "algerian_law"
VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore"
DEFAULT_TOP_K = 5

# ── Lazy-initialized singletons (created on first use, not at import time) ────
_genai_client = None
_chroma_collection = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set. Check your .env file.")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _get_collection():
    """Load the persistent ChromaDB collection (lazy singleton)."""
    global _chroma_collection
    if _chroma_collection is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_DIR}.\n"
                "Run `python backend/rag/embedding/embed_articles.py` first."
            )
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


def embed_query(query: str) -> List[float]:
    """Embed the user query for similarity search."""
    response = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles for the given query.

    Args:
        query:  The user's natural-language question (Arabic or French)
        top_k:  Number of results to return (default 5)

    Returns:
        List of dicts with: id, text, metadata, score
    """
    normalized_query = normalize_arabic(query)
    query_embedding = embed_query(normalized_query)

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc_id, doc_text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "id": doc_id,
            "text": doc_text,
            "metadata": meta,
            "score": round(dist, 4),
        })

    return retrieved


if __name__ == "__main__":
    test_query = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"Result {i} [score={r['score']}]")
        print(f"  ID     : {r['id']}")
        print(f"  Law    : {r['metadata'].get('law_name', '')}")
        print(f"  Article: {r['metadata'].get('article_number', '')}")
        print(f"  Title  : {r['metadata'].get('title', '')}")
        print(f"  Text   : {r['metadata'].get('text_original', '')[:200]}...")
        print()
