import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from hybrid_rag.hybrid_retriever import bm25_retrieve, dense_retrieve, reciprocal_rank_fusion

from hybrid_rag.reranker import rerank_candidates

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

q = "ما هي عقوبة الغش في بيع السلع؟"
print(f"Query: {q}")



bm25 = bm25_retrieve(q, top_k=20)
dense = dense_retrieve(q, top_k=20)

fused = reciprocal_rank_fusion(bm25, dense)

print("\n--- Top 10 Fused (Pre-Rerank) ---")
for i, r in enumerate(fused[:10], 1):
    print(f"[{i}] {r.get('id', 'N/A')} | {r['law_name']} - {r['article_number']} (Score: {r['rrf_score']})")

reranked = rerank_candidates(q, fused, top_k=10)
print("\n--- Top 10 Reranked ---")
for i, r in enumerate(reranked, 1):
    print(f"[{i}] {r.get('id', 'N/A')} | {r['law_name']} - {r['article_number']} (Score: {r['rerank_score']})")

