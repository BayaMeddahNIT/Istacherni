"""
test_graph_rag.py
-----------------
End-to-end test for the Graph RAG pipeline.

What makes Graph RAG different:
  • No API needed to BUILD the knowledge graph (pure JSON metadata)
  • Retrieval uses graph traversal (concept → article → related-article)
  • Results are ranked by graph centrality (PageRank) + hit count
  • The path through the graph is visible and explainable

Usage:
  python test_graph_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_rag.graph_retriever import graph_retrieve
from graph_rag.graph_generator  import graph_generate

# ─── Config ───────────────────────────────────────────────────────────────────
USE_GENERATION = True
TOP_K          = 5

TEST_QUERIES = [
    # Penal Law (قانون العقوبات)
    "ما هي عقوبة الغش في بيع السلع؟",
    "ما هي عقوبة إصدار شيك بدون رصيد؟",
    "هل الاحتيال الإلكتروني (online scam) جريمة؟",
    # Civil Law (القانون المدني)
    "هل العقد الشفهي ملزم قانونياً؟",
    "ما هي شروط صحة العقد؟",
    "ما هي القوة القاهرة في العقود؟",
    # Administrative Law (القانون الإداري)
    "كيف أتحصل على رخصة تجارية؟",
    "هل يمكن مقاضاة إدارة عمومية؟",
    "كيف أستخرج سجل تجاري؟",
    # Labor Law (قانون العمل)
    "ما هي حقوق العامل في الجزائر؟",
    "هل يمكن طردي بدون سبب؟",
    "ما هي ساعات العمل القانونية؟",
    # Commercial Law (القانون التجاري)
    "كيف أفتح شركة في الجزائر؟",
    "ما الفرق بين SARL و SPA؟",
    "ما هي إجراءات الإفلاس؟",
]

# ─── Runner ───────────────────────────────────────────────────────────────────
def run(query: str):
    print(f"\n{'='*68}")
    print(f"  QUERY : {query}")
    print(f"{'='*68}")

    # ── Step 1: Graph Retrieval ────────────────────────────────────────
    print("\n🕸️  Step 1 — Knowledge Graph Traversal")
    results = graph_retrieve(query, top_k=TOP_K)

    if not results:
        print("  ❌ No articles found in graph.")
        return

    print(f"  Retrieved {len(results)} articles:")
    for r in results:
        print(
            f"    ├─ [{r['law_name']}] المادة {r['article_number']:6s}  "
            f"graph={r['graph_score']:.3f}  pr={r['pagerank']:.6f}"
            f"  | {r['title'][:45]}"
        )

    # ── Step 2: Generation ────────────────────────────────────────────
    if not USE_GENERATION:
        print("\n  ⚠️  Generation skipped.")
        return

    print("\n🤖 Step 2 — Answer Generation (Gemini 2.5 Flash)")
    try:
        answer = graph_generate(query, results)
        print("\n  ✅ Answer:")
        print("  " + "─" * 62)
        for line in answer.splitlines():
            print(f"  {line}")
        print("  " + "─" * 62)
    except Exception as e:
        print(f"\n  ❌ Generation failed: {e}")


if __name__ == "__main__":
    print("\n" + "="*68)
    print("   GRAPH RAG — End-to-End Test")
    print("="*68)

    for q in TEST_QUERIES:
        run(q)

    print("\n\n🏁 All tests complete.")
