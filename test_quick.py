import sys
from pathlib import Path
from graph_rag_local.graph_retriever import graph_retrieve
from graph_rag_local.graph_generator  import graph_generate

QUERIES = [
    "ما هي عقوبة الغش في بيع السلع؟",  # Penal Law
    "هل يمكن مقاضاة إدارة عمومية؟",     # Admin Law
]

def run_quick_test():
    for q in QUERIES:
        print(f"\nQUERY: {q}")
        print("-" * 60)
        results = graph_retrieve(q, top_k=5)
        for r in results:
            print(f"   ├─ [{r['law_name']}] المادة {r['article_number']}  score={r['graph_score']:.3f} | {r.get('title','')[:40]}")
        
        print("\n🤖 Generating answer...")
        answer = graph_generate(q, results, stream=False)
        print(f"\n✅ Answer:\n{answer}\n")
        print("=" * 70)

if __name__ == "__main__":
    run_quick_test()
