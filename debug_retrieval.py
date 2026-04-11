# debug_retrieval.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from bm25_rag.bm25_retriever import bm25_retrieve
from dense_rag.bge_retriever import dense_retrieve

q = "ما هي عقوبة السرقة في القانون الجزائري؟"

print("=== BM25 Results ===")
for r in bm25_retrieve(q, top_k=5):
    print(f"  id={r.get('id')} | {r['law_name']} م{r['article_number']} | score={r['score']}")

print("\n=== Dense Results ===")
for r in dense_retrieve(q, top_k=5):
    print(f"  id={r.get('id')} | {r['law_name']} م{r['article_number']} | score={r['score']}")