"""
faiss_retriever.py
------------------
Retrieval module using FAISS — a parallel alternative to retriever.py (ChromaDB).
Your existing ChromaDB retriever is NOT modified.

This module exposes:
  - retrieve_faiss(query, top_k)          → pure FAISS results
  - retrieve_hybrid(query, top_k, alpha)  → RRF fusion of FAISS + ChromaDB results

Usage (standalone):
  python backend/rag/retrieval/faiss_retriever.py

Usage (in your pipeline):
  from backend.rag.retrieval.faiss_retriever import retrieve_faiss, retrieve_hybrid
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import faiss
from google import genai
from google.genai import types as genai_types

from backend.rag.preprocessing.normalize_arabic import normalize_arabic

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL  = "models/gemini-embedding-001"
FAISS_STORE_DIR  = PROJECT_ROOT / "backend" / "faiss_store"
INDEX_FILE       = FAISS_STORE_DIR / "algerian_law.index"
METADATA_FILE    = FAISS_STORE_DIR / "algerian_law_metadata.pkl"
DEFAULT_TOP_K    = 5

# ── Lazy singletons ────────────────────────────────────────────────────────────
_genai_client   = None
_faiss_index    = None
_metadata_list  = None   # list[dict] aligned to FAISS integer IDs


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set. Check your .env file.")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _get_index_and_metadata():
    """Lazy-load the FAISS index and metadata list from disk."""
    global _faiss_index, _metadata_list
    if _faiss_index is None:
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_FILE}.\n"
                "Run `python backend/rag/embedding/faiss_indexer.py` first."
            )
        _faiss_index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, "rb") as f:
            _metadata_list = pickle.load(f)
    return _faiss_index, _metadata_list


# ── Core retrieval ─────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a (1, D) float32 array, L2-normalized (matches indexer normalization).
    """
    response = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    vec = np.array([response.embeddings[0].values], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def retrieve_faiss(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles using FAISS.

    Args:
        query:  User question (Arabic or French).
        top_k:  Number of results to return.

    Returns:
        List of dicts with keys: id, text, metadata, score
        `score` is the cosine similarity (higher = more relevant, range 0–1).
    """
    normalized_query = normalize_arabic(query)
    query_vec = embed_query(normalized_query)

    index, metadata_list = _get_index_and_metadata()
    k = min(top_k, index.ntotal)

    # scores shape: (1, k),  ids shape: (1, k)
    scores, ids = index.search(query_vec, k)

    results = []
    for score, faiss_id in zip(scores[0], ids[0]):
        if faiss_id == -1:
            # FAISS returns -1 when there are fewer results than requested
            continue
        meta = metadata_list[int(faiss_id)]
        results.append({
            "id":       meta.get("id", str(faiss_id)),
            "text":     meta.get("text", ""),
            "metadata": {k: v for k, v in meta.items() if k not in ("id", "text")},
            "score":    round(float(score), 4),   # cosine similarity
        })

    return results


# ── Hybrid retrieval (FAISS + ChromaDB via Reciprocal Rank Fusion) ─────────────

def retrieve_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining FAISS and ChromaDB results using
    Reciprocal Rank Fusion (RRF).

    Both retrievers independently return `top_k * 2` candidates;
    their ranks are fused with RRF, and the top `top_k` are returned.

    Args:
        query:  User question.
        top_k:  Final number of results to return.
        alpha:  Weight for FAISS vs ChromaDB.
                alpha=1.0 → pure FAISS, alpha=0.0 → pure ChromaDB.
                Default 0.5 = equal weight.
        rrf_k:  RRF smoothing constant (default 60, standard value).

    Returns:
        List of merged result dicts, sorted by fused score descending.
    """
    # Import here to avoid circular imports at module load time
    from backend.rag.retrieval.retriever import retrieve as chroma_retrieve

    fetch_n = top_k * 2   # fetch more candidates before fusion

    faiss_results  = retrieve_faiss(query, top_k=fetch_n)
    chroma_results = chroma_retrieve(query, top_k=fetch_n)

    # Map doc_id → result dict (FAISS takes priority for text/metadata)
    doc_store: Dict[str, Dict] = {}
    for r in faiss_results:
        doc_store[r["id"]] = r
    for r in chroma_results:
        if r["id"] not in doc_store:
            doc_store[r["id"]] = r

    # RRF score accumulator
    rrf_scores: Dict[str, float] = {}

    def _rrf(rank: int) -> float:
        return 1.0 / (rrf_k + rank + 1)

    # FAISS contribution (weighted by alpha)
    for rank, result in enumerate(faiss_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + alpha * _rrf(rank)

    # ChromaDB contribution (weighted by 1 - alpha)
    chroma_weight = 1.0 - alpha
    for rank, result in enumerate(chroma_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + chroma_weight * _rrf(rank)

    # Sort by fused score, return top_k
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    fused_results = []
    for doc_id in sorted_ids:
        result = doc_store[doc_id].copy()
        result["score"] = round(rrf_scores[doc_id], 6)
        result["score_type"] = "rrf_fused"
        fused_results.append(result)

    return fused_results


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Query: {test_query}\n")

    print("─" * 50)
    print("FAISS results:")
    print("─" * 50)
    faiss_results = retrieve_faiss(test_query, top_k=3)
    for i, r in enumerate(faiss_results, 1):
        print(f"Result {i} [cosine={r['score']}]")
        print(f"  ID      : {r['id']}")
        print(f"  Law     : {r['metadata'].get('law_name', '')}")
        print(f"  Article : {r['metadata'].get('article_number', '')}")
        print(f"  Text    : {r['metadata'].get('text_original', r['text'])[:200]}...")
        print()

    print("─" * 50)
    print("Hybrid results (FAISS + ChromaDB, RRF fusion):")
    print("─" * 50)
    hybrid_results = retrieve_hybrid(test_query, top_k=3)
    for i, r in enumerate(hybrid_results, 1):
        print(f"Result {i} [rrf_score={r['score']}]")
        print(f"  ID      : {r['id']}")
        print(f"  Law     : {r['metadata'].get('law_name', '')}")
        print(f"  Article : {r['metadata'].get('article_number', '')}")
        print()