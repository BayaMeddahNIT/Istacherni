"""
local_retriever.py
------------------
Retrieval module — local embedding variant.
Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 to embed
the user query, then queries the local ChromaDB collection built by
embed_articles_local.py.

The embedding model MUST match the one used during ingestion:
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Usage (from generator.py or any other module):
  from backend.rag.retrieval.local_retriever import retrieve
  chunks = retrieve("ما هي عقوبة السرقة؟", top_k=5)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from backend.rag.preprocessing.normalize_arabic import normalize_arabic

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME      = "algerian_law_local"
VECTORSTORE_DIR      = PROJECT_ROOT / "backend" / "vectorstore_local"
DEFAULT_TOP_K        = 5

# ── Lazy-initialized singletons (created on first use, not at import time) ────
_embedding_model   = None
_chroma_collection = None


def _get_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformers model (lazy singleton)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection():
    """Load the persistent ChromaDB collection (lazy singleton)."""
    global _chroma_collection
    if _chroma_collection is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_DIR}.\n"
                "Run `python backend/rag/embedding/embed_articles_local.py` first."
            )
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string using the local model.
    Unit-normalised so cosine similarity == dot product.
    """
    vector = _get_embedding_model().encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles for the given query.

    Args:
        query:  The user's natural-language question (Arabic or French)
        top_k:  Number of results to return (default 5)

    Returns:
        List of dicts with keys: id, text, metadata, score
        'score' is the cosine distance (lower = more similar).
    """
    normalized_query = normalize_arabic(query)
    query_embedding  = embed_query(normalized_query)

    collection = _get_collection()
    results    = collection.query(
        query_embeddings = [query_embedding],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas", "distances"],
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
        print(f"Result {i} [score={r['score']}]")
        print(f"  ID     : {r['id']}")
        print(f"  Law    : {r['metadata'].get('law_name', '')}")
        print(f"  Article: {r['metadata'].get('article_number', '')}")
        print(f"  Text   : {r['metadata'].get('text_original', '')[:200]}...")
        print()