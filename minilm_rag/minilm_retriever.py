"""
minilm_retriever.py
-------------------
Hybrid retrieval: BM25 (keyword) + MiniLM dense (semantic).

Scores are independently min-max normalised to [0, 1], then linearly combined:
    hybrid_score = BM25_WEIGHT * bm25_norm + EMBED_WEIGHT * cosine_norm

Default weights: 0.4 BM25 + 0.6 semantic — tunable via retrieve(…, bm25_weight=…).

Usage (standalone):
  python -m minilm_rag.minilm_retriever

Programmatic:
  from minilm_rag.minilm_retriever import minilm_retrieve
  results = minilm_retrieve("ما هي عقوبة السرقة؟", top_k=5)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from minilm_rag.minilm_indexer import (
    EMBED_MODEL_NAME,
    build_index,
    tokenize_arabic,
)

# ── Default fusion weights ─────────────────────────────────────────────────────
DEFAULT_BM25_WEIGHT  = 0.4
DEFAULT_EMBED_WEIGHT = 0.6

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_indexes: dict | None = None
_model:   SentenceTransformer | None = None


def _get_indexes() -> dict:
    global _indexes
    if _indexes is None:
        _indexes = build_index()   # loads from cache if already built
    return _indexes


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[MiniLM-Retriever] Loading model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


# ── Score normalisation ────────────────────────────────────────────────────────
def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros if all values are equal."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ── Main retrieval function ────────────────────────────────────────────────────
def minilm_retrieve(
    query:       str,
    top_k:       int   = 10,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    embed_weight: float = DEFAULT_EMBED_WEIGHT,
    pre_filter:  int   = 200,   # BM25 candidate pool before re-ranking
) -> list[dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles using hybrid BM25 + MiniLM.

    Args:
        query:        User's question (Arabic / French / mixed).
        top_k:        Number of final results.
        bm25_weight:  Weight for BM25 scores in fusion (default 0.4).
        embed_weight: Weight for cosine-sim scores in fusion (default 0.6).
        pre_filter:   Number of BM25 candidates to pass to dense re-ranking.
                      Larger → better recall, slower. (default 200)

    Returns:
        List of dicts, each containing:
          id, law_name, law_domain, article_number, title,
          text_original, penalties_summary, legal_conditions_summary,
          keywords, bm25_score, embed_score, hybrid_score
    """
    if not query.strip():
        return []

    idx   = _get_indexes()
    model = _get_model()
    bm25         = idx["bm25"]
    corpus       = idx["bm25_corpus"]      # same order as embeddings
    all_embeddings = idx["embeddings"]     # shape (N, D), already unit-normed

    N = len(corpus)

    # ── Step 1 · BM25 scores over full corpus ─────────────────────────────────
    tokens     = tokenize_arabic(query)
    bm25_scores = bm25.get_scores(tokens) if tokens else np.zeros(N)

    # ── Step 2 · Dense cosine similarity ──────────────────────────────────────
    query_vec = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )                                      # shape (D,)
    cosine_scores = all_embeddings @ query_vec  # shape (N,)  (dot = cosine since normed)

    # ── Step 3 · Fusion ───────────────────────────────────────────────────────
    bm25_norm   = _minmax(bm25_scores)
    cosine_norm = _minmax(cosine_scores)

    hybrid = bm25_weight * bm25_norm + embed_weight * cosine_norm

    # ── Step 4 · Top-K ────────────────────────────────────────────────────────
    top_indices = np.argsort(hybrid)[::-1][:top_k]

    results: list[dict] = []
    for idx_i in top_indices:
        score = float(hybrid[idx_i])
        if score <= 0:
            continue
        art = corpus[idx_i]
        results.append({
            "id":                       art["id"],
            "law_name":                 art["law_name"],
            "law_domain":               art["law_domain"],
            "article_number":           art["article_number"],
            "title":                    art["title"],
            "text_original":            art["text_original"],
            "penalties_summary":        art.get("penalties_summary", ""),
            "legal_conditions_summary": art.get("legal_conditions_summary", ""),
            "keywords":                 art.get("keywords", []),
            # ── individual scores for debugging / tuning ──
            "bm25_score":   round(float(bm25_scores[idx_i]),   4),
            "embed_score":  round(float(cosine_scores[idx_i]), 4),
            "hybrid_score": round(score,                        4),
        })

    return results


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
        "التهديد بالعنف",
    ]

    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {q}")
        print("=" * 65)
        hits = minilm_retrieve(q, top_k=3)
        if not hits:
            print("  No results.")
            continue
        for i, r in enumerate(hits, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Title      : {r['title']}")
            print(f"       BM25       : {r['bm25_score']}  "
                  f"Embed: {r['embed_score']}  "
                  f"Hybrid: {r['hybrid_score']}")
            print(f"       Text       : {r['text_original'][:140]}…")
