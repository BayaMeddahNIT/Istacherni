"""
bge_retriever.py
----------------
ChromaDB RETRIEVAL using local BAAI/bge-m3 embeddings.
Drop-in replacement for retriever.py (Gemini).
Reads from the collection written by bge_embed_articles.py.

Usage (standalone):
  python backend/rag/retrieval/bge_retriever.py

Usage (in pipeline):
  from backend.rag.retrieval.bge_retriever import retrieve
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

from backend.rag.preprocessing.normalize_arabic import normalize_arabic
from backend.rag.embedding.bge_embedder import embed_query as _embed_query

# ── Configuration ─────────────────────────────────────────────────────────────
COLLECTION_NAME = "algerian_law_bge"
VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore"
DEFAULT_TOP_K   = 5

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_chroma_collection = None


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_DIR}.\n"
                "Run `python backend/rag/embedding/bge_embed_articles.py` first."
            )
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles for the given query
    using local BGE-m3 embeddings.

    Args:
        query:  User question (Arabic, French, or mixed).
        top_k:  Number of results to return.

    Returns:
        List of dicts with keys: id, text, metadata, score
        `score` is the cosine distance (lower = more similar, ChromaDB default).
    """
    normalized_query  = normalize_arabic(query)
    # embed_query returns shape (1, 1024); ChromaDB needs a plain list
    query_embedding   = _embed_query(normalized_query)[0].tolist()

    collection = _get_collection()
    results    = collection.query(
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
            "id":       doc_id,
            "text":     doc_text,
            "metadata": meta,
            "score":    round(dist, 4),
        })

    return retrieved


if __name__ == "__main__":
    test_query = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"Result {i}  [score={r['score']}]")
        print(f"  ID      : {r['id']}")
        print(f"  Law     : {r['metadata'].get('law_name', '')}")
        print(f"  Article : {r['metadata'].get('article_number', '')}")
        print(f"  Text    : {r['metadata'].get('text_original', '')[:200]} …")
        print()
