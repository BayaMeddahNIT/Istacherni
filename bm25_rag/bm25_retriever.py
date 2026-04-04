"""
bm25_retriever.py
-----------------
BM25 retrieval module. Completely standalone — uses no ChromaDB, no embeddings.

Usage (standalone):
  python bm25_rag/bm25_retriever.py

Programmatic:
  from bm25_rag.bm25_retriever import bm25_retrieve
  results = bm25_retrieve("ما هي عقوبة السرقة؟", top_k=5)
"""

import re
from pathlib import Path
from typing import List, Dict, Any

from bm25_rag.bm25_indexer import build_index, tokenize_arabic

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_bm25 = None
_corpus: list = []


def _get_index():
    global _bm25, _corpus
    if _bm25 is None:
        _bm25, _corpus = build_index()   # loads from cache if already built
    return _bm25, _corpus


def bm25_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles for a query using BM25.

    Args:
        query:  User's natural-language question (Arabic or French).
        top_k:  Number of results.

    Returns:
        List of dicts keyed: id, law_name, law_domain, article_number,
                             title, text_original, score
    """
    bm25, corpus = _get_index()

    tokens = tokenize_arabic(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    # Get top_k indices sorted by descending score
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        article = corpus[idx]
        score   = float(scores[idx])
        if score <= 0:
            continue
        results.append({
            "id":             article["id"],
            "law_name":       article["law_name"],
            "law_domain":     article["law_domain"],
            "article_number": article["article_number"],
            "title":          article["title"],
            "text_original":  article["text_original"],
            "penalties_summary":        article.get("penalties_summary", ""),
            "legal_conditions_summary": article.get("legal_conditions_summary", ""),
            "keywords":       article.get("keywords", []),
            "score":          round(score, 4),
        })

    return results


if __name__ == "__main__":
    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        results = bm25_retrieve(q, top_k=3)
        if not results:
            print("  No results found.")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Title : {r['title']}")
            print(f"       Score : {r['score']}")
            print(f"       Text  : {r['text_original'][:150]}…")
