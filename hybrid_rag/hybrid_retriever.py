# hybrid_rag/hybrid_retriever.py
"""
Hybrid retrieval: BM25 (keyword) + BGE-M3 (semantic) → RRF fusion.
This is the main retriever to use in production.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bm25_rag.bm25_retriever import bm25_retrieve
from dense_rag.bge_retriever import dense_retrieve


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,           # RRF constant — 60 is standard
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = Σ weight / (k + rank)
    Higher score = more relevant.
    """
    scores: dict[str, float] = {}
    article_map: dict[str, dict] = {}

    for rank, article in enumerate(bm25_results, start=1):
        aid = article["id"] if "id" in article else f"{article['law_name']}_{article['article_number']}"
        scores[aid] = scores.get(aid, 0) + bm25_weight / (k + rank)
        article_map[aid] = article

    for rank, article in enumerate(dense_results, start=1):
        aid = article.get("id") or f"{article['law_name']}_{article['article_number']}"
        scores[aid] = scores.get(aid, 0) + dense_weight / (k + rank)
        if aid not in article_map:
            article_map[aid] = article

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    
    fused = []
    for aid in sorted_ids:
        article = article_map[aid].copy()
        article["rrf_score"] = round(scores[aid], 6)
        article["retrieval_method"] = "hybrid"
        fused.append(article)
    
    return fused


def hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Main retrieval function for production use.
    Fetches more candidates from each method, then fuses and returns top_k.
    """
    # Fetch more candidates than needed before fusion
    fetch_k = top_k * 3

    bm25_results  = bm25_retrieve(query, top_k=fetch_k)
    dense_results = dense_retrieve(query, top_k=fetch_k)
    
    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    return fused[:top_k]


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
        "عقوبة تزوير الوثائق الرسمية",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results = hybrid_retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       RRF Score : {r['rrf_score']}")
            print(f"       Title     : {r.get('title', '')[:80]}")