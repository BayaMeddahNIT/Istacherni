"""
qwen_retriever.py
-----------------
Retrieval module: given a user query, embed it locally with
Qwen3-Embedding-0.6B and find the top-K most relevant law article
chunks from the Qwen-based ChromaDB collection.

Drop-in replacement for retriever.py — same return schema:
  [{"id": str, "text": str, "metadata": dict, "score": float}, ...]

Usage (standalone test):
  cd <project root>
  python backend/rag/retrieval/qwen_retriever.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# ── Project root on sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import chromadb
from sentence_transformers import SentenceTransformer

from backend.rag.preprocessing.normalize_arabic import normalize_arabic

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "algerian_law_qwen"
VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore_qwen"
DEFAULT_TOP_K   = 5

# Asymmetric retrieval: the query uses a different instruction than the passage.
# This matches what qwen_embed_articles.py uses for documents.
QUERY_PROMPT = (
    "Instruct: Given a legal question, retrieve the most relevant legal articles.\nQuery: "
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embedding_model    = None
_chroma_collection  = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading embedding model '{MODEL_NAME}' on {device}…")
        _embedding_model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True,
            device=device,
        )
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    """Open the persistent ChromaDB collection once and cache it."""
    global _chroma_collection
    if _chroma_collection is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_DIR}.\n"
                "Run `python backend/rag/embedding/qwen_embed_articles.py` first."
            )
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string using the Qwen3 query instruction prompt.

    The QUERY_PROMPT prefix signals to the model that this is a question
    looking for relevant passages (asymmetric retrieval).
    """
    prompted = QUERY_PROMPT + query
    vector = _get_model().encode(
        prompted,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles for the given query.

    Args:
        query:  The user's natural-language question (Arabic or French).
        top_k:  Number of results to return (default 5).
        where:  Optional ChromaDB metadata filter, e.g.
                {"law_domain": "Penal Law"}.

    Returns:
        List of dicts, each containing:
          - id       : chunk ID (e.g. "DZ_PENAL_ART_350_01")
          - text     : normalized embeddable text
          - metadata : raw metadata dict from ChromaDB
          - score    : cosine distance (lower = more similar)
    """
    normalized_query  = normalize_arabic(query)
    query_embedding   = embed_query(normalized_query)
    collection        = _get_collection()

    query_kwargs: Dict[str, Any] = dict(
        query_embeddings = [query_embedding],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

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
            "score":    round(float(dist), 4),
        })

    return retrieved


if __name__ == "__main__":
    test_query = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Query: {test_query}\n")

    results = retrieve(test_query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"Result {i}  [score={r['score']}]")
        print(f"  ID     : {r['id']}")
        print(f"  Law    : {r['metadata'].get('law_name', '')}")
        print(f"  Article: {r['metadata'].get('article_number', '')}")
        print(f"  Text   : {r['metadata'].get('text_original', r['text'])[:200]}…")
        print()