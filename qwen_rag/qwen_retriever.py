"""
qwen_retriever.py
-----------------
Dense retrieval using Qwen3-Embedding-0.6B + ChromaDB.
Drop-in replacement for bm25_retrieve() — same return schema.

Usage (standalone):
    python qwen_rag/qwen_retriever.py

Programmatic:
    from qwen_rag.qwen_retriever import qwen_retrieve
    results = qwen_retrieve("ما هي عقوبة السرقة؟", top_k=5)
"""

from __future__ import annotations

# ── Make "qwen_rag" importable when run as a plain script ─────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────

import json
from typing import Any, Dict, List

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        from qwen_rag.qwen_indexer import load_collection
        _collection = load_collection()
    return _collection


# ── Public API ─────────────────────────────────────────────────────────────────

def qwen_retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most semantically similar law articles for a query.

    Args:
        query:      User's natural-language question (Arabic preferred).
        top_k:      Number of results to return.
        min_score:  Minimum cosine similarity threshold (0–1).

    Returns:
        List of article dicts sorted by descending similarity, each containing:
        id, law_name, law_domain, article_number, title, text_original,
        penalties_summary, legal_conditions_summary, keywords (list), score.
    """
    from qwen_rag.qwen_embedder import embed_query

    collection = _get_collection()

    q_vec = embed_query(query).tolist()[0]

    # Fetch a wider set so we can apply min_score filter afterwards
    n_results = min(top_k * 3, collection.count())
    response  = collection.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results   = []
    ids       = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance = 1 − cosine_sim  → invert back
        similarity = round(1.0 - dist, 4)

        if similarity < min_score:
            continue

        results.append({
            "id":                       doc_id,
            "law_name":                 meta.get("law_name", ""),
            "law_domain":               meta.get("law_domain", ""),
            "article_number":           meta.get("article_number", ""),
            "title":                    meta.get("title", ""),
            "text_original":            text,
            "penalties_summary":        meta.get("penalties_summary", ""),
            "legal_conditions_summary": meta.get("legal_conditions_summary", ""),
            "keywords":                 json.loads(meta.get("keywords", "[]")),
            "score":                    similarity,
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
        "عقوبة غش المواد الغذائية",
    ]

    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {q}")
        print("=" * 65)
        hits = qwen_retrieve(q, top_k=3)
        if not hits:
            print("  No results found.")
        for i, r in enumerate(hits, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Title : {r['title']}")
            print(f"       Score : {r['score']}")
            print(f"       Text  : {r['text_original'][:150]}…")
