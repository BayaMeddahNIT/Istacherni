# hybrid_rag/hybrid_retriever.py
"""
Hybrid retrieval: BM25 (keyword) + BGE-M3 (semantic) → Weighted Linear Combination.
This is the main retriever to use in production.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bm25_rag.bm25_retriever import bm25_retrieve
from dense_rag.bge_retriever import dense_retrieve
from hybrid_rag.reranker import rerank_candidates


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return {}
    min_val = min(scores.values())
    max_val = max(scores.values())
    
    if max_val == min_val:
        return {k: 1.0 for k in scores}
        
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def weighted_linear_combination(
    bm25_results: list[dict],
    dense_results: list[dict],
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
) -> list[dict]:
    """
    Merge two ranked lists using Weighted Linear Combination.
    Score = alpha * DenseScore + (1 - alpha) * BM25Score.
    Scores are Min-Max normalized before combination.
    """
    bm25_scores = {}
    dense_scores = {}
    article_map = {}

    # Extract raw scores
    for article in bm25_results:
        aid = article.get("id") or f"{article['law_name']}_{article['article_number']}"
        bm25_scores[aid] = article.get("score", 0.0)
        article_map[aid] = article

    for article in dense_results:
        aid = article.get("id") or f"{article['law_name']}_{article['article_number']}"
        dense_scores[aid] = article.get("score", 0.0)
        if aid not in article_map:
            article_map[aid] = article

    # Normalize scores
    norm_bm25 = min_max_normalize(bm25_scores)
    norm_dense = min_max_normalize(dense_scores)

    # Combine scores
    combined_scores = {}
    for aid in article_map:
        score_bm25 = norm_bm25.get(aid, 0.0)
        score_dense = norm_dense.get(aid, 0.0)
        combined_scores[aid] = (bm25_weight * score_bm25) + (dense_weight * score_dense)

    # Sort by combined score descending
    sorted_ids = sorted(combined_scores, key=lambda x: combined_scores[x], reverse=True)
    
    fused = []
    for aid in sorted_ids:
        article = article_map[aid].copy()
        article["fusion_score"] = round(combined_scores[aid], 6)
        article["norm_bm25"] = round(norm_bm25.get(aid, 0.0), 6)
        article["norm_dense"] = round(norm_dense.get(aid, 0.0), 6)
        article["retrieval_method"] = "hybrid"
        fused.append(article)
    
    return fused


def hybrid_retrieve(query: str, top_k: int = 5, bm25_weight: float = 0.3, dense_weight: float = 0.7) -> list[dict]:
    """
    Main retrieval function for production use.
    Fetches candidates from each method, fuses them with given weights, and reranks.
    """
    fetch_k = 100
    bm25_results = bm25_retrieve(query, top_k=fetch_k)
    dense_results = dense_retrieve(query, top_k=fetch_k)
    fused = weighted_linear_combination(bm25_results, dense_results, bm25_weight=bm25_weight, dense_weight=dense_weight)
    reranked = rerank_candidates(query, fused[:100], top_k=top_k)
    return reranked


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    test_queries = [
        "هل يمكن للشريك في شركة SNC أن يتنازل عن حصصه لزوجته دون موافقة البقية؟"
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results = hybrid_retrieve(q, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Rerank Score: {r.get('rerank_score', 'N/A')} (Fusion: {r.get('fusion_score', 'N/A')})")
            print(f"       Title       : {r.get('title', '')[:80]}")