"""
hybrid_retriever.py
-------------------
Hybrid retrieval combining BM25 (sparse, keyword-based) and CAMeLBERT
(dense, semantic) scores via Reciprocal Rank Fusion (RRF).

RRF formula (Cormack et al., 2009):
    score(d) = Σ  1 / (k + rank_i(d))
  where k=60 is the standard smoothing constant and rank_i(d) is the
  1-based rank of document d in retrieval list i.

Why RRF instead of simple score interpolation?
  • Score scales are incomparable (BM25 is unbounded; cosine is 0–1).
  • RRF is parameter-free beyond k and consistently matches or beats
    learned interpolation in information-retrieval benchmarks.
  • Robust: a single very-high BM25 score cannot dominate.

Usage:
  from camelbert_rag.hybrid_retriever import hybrid_retrieve
  results = hybrid_retrieve("ما هي عقوبة السرقة؟", top_k=5)
"""

from __future__ import annotations

from typing import Any

from bm25_rag.bm25_retriever import bm25_retrieve
from camelbert_rag.camelbert_retriever import camelbert_retrieve

# ── RRF constant ───────────────────────────────────────────────────────────────
_RRF_K = 60


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Return Reciprocal Rank Fusion score for 1-based rank."""
    return 1.0 / (k + rank)


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    bm25_candidates: int = 20,
    dense_candidates: int = 20,
) -> list[dict[str, Any]]:
    """
    Fuse BM25 and CAMeLBERT retrieval results using Reciprocal Rank Fusion.

    Args:
        query:            User's question (Arabic).
        top_k:            Number of final results to return.
        bm25_candidates:  How many candidates to pull from BM25 before fusion.
        dense_candidates: How many candidates to pull from CAMeLBERT before fusion.

    Returns:
        List of article dicts (top_k), sorted by descending RRF score, each
        containing all standard fields plus:
          'rrf_score'    – the fused RRF score
          'bm25_rank'    – rank in the BM25 list (None if absent)
          'dense_rank'   – rank in the CAMeLBERT list (None if absent)
    """
    # ── Retrieve from both sources ─────────────────────────────────────────────
    bm25_hits   = bm25_retrieve(query,       top_k=bm25_candidates)
    dense_hits  = camelbert_retrieve(query,  top_k=dense_candidates)

    # ── Build ID → article lookup and RRF score accumulator ───────────────────
    doc_map:   dict[str, dict]  = {}   # doc_id → article dict
    rrf_scores: dict[str, float] = {}  # doc_id → accumulated RRF score
    bm25_ranks: dict[str, int]   = {}
    dense_ranks: dict[str, int]  = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["id"]
        doc_map[doc_id]   = hit
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + _rrf_score(rank)
        bm25_ranks[doc_id] = rank

    for rank, hit in enumerate(dense_hits, start=1):
        doc_id = hit["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = hit
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + _rrf_score(rank)
        dense_ranks[doc_id] = rank

    # ── Sort by RRF score and take top_k ────────────────────────────────────────
    ranked_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    results = []
    for doc_id in ranked_ids:
        article = dict(doc_map[doc_id])        # shallow copy so we don't mutate cache
        article["rrf_score"]  = round(rrf_scores[doc_id], 6)
        article["bm25_rank"]  = bm25_ranks.get(doc_id)
        article["dense_rank"] = dense_ranks.get(doc_id)
        results.append(article)

    return results


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "عقوبة تزوير الوثائق الرسمية",
        "حقوق العامل عند الفصل التعسفي",
    ]

    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"Hybrid Query: {q}")
        print(f"{'='*65}")
        hits = hybrid_retrieve(q, top_k=5)
        for i, r in enumerate(hits, 1):
            b = r["bm25_rank"]
            d = r["dense_rank"]
            print(
                f"  [{i}] RRF={r['rrf_score']:.5f}  "
                f"BM25={b or '—':>3}  Dense={d or '—':>3}  "
                f"{r['law_name']} — المادة {r['article_number']}"
            )
            print(f"       {r['title']}")