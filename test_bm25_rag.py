"""
test_bm25_rag.py
----------------
End-to-end test for the standalone BM25 RAG pipeline.

Steps:
  1. Build (or load) the BM25 index
  2. Retrieve top-K articles for several test queries
  3. Generate Arabic answers using Gemini

Usage:
  python test_bm25_rag.py
"""

import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bm25_rag.bm25_retriever import bm25_retrieve
from bm25_rag.bm25_generator  import bm25_generate

# ─── Configuration ────────────────────────────────────────────────────────────
USE_GENERATION = True   # Set to False to only test retrieval (saves API calls)
TOP_K          = 5

TEST_QUERIES = [
    # Penal Law (قانون العقوبات)

    #"ما هي عقوبة الغش في بيع السلع؟",
    #"ما هي عقوبة إصدار شيك بدون رصيد؟",
    #"هل الاحتيال الإلكتروني (online scam) جريمة؟",

    # Civil Law (القانون المدني)

    #"هل العقد الشفهي ملزم قانونياً؟",
    "ما هي شروط صحة العقد؟",
    "ما هي القوة القاهرة في العقود؟",

    # Administrative Law (القانون الإداري)

    #"كيف أتحصل على رخصة تجارية؟",
    #"هل يمكن مقاضاة إدارة عمومية؟",
    #"كيف أستخرج سجل تجاري؟",

    # Labor Law (قانون العمل)

    #"ما هي حقوق العامل في الجزائر؟",
    #"هل يمكن طردي بدون سبب؟",
    #"ما هي ساعات العمل القانونية؟",

    # Commercial Law (القانون التجاري)
    
    #"كيف أفتح شركة في الجزائر؟",
    #"ما الفرق بين SARL و SPA؟",
    #"ما هي إجراءات الإفلاس؟",
]

# ─── Main ─────────────────────────────────────────────────────────────────────
def run_test(query: str):
    print(f"\n{'='*65}")
    print(f"  QUERY: {query}")
    print(f"{'='*65}")

    # Step 1: Retrieval
    print("\n🔎 Step 1 — BM25 Retrieval")
    results = bm25_retrieve(query, top_k=TOP_K)

    if not results:
        print("  ❌ No articles retrieved.")
        return

    print(f"  Retrieved {len(results)} articles:")
    for r in results:
        print(f"    ├─ [{r['law_name']}] المادة {r['article_number']:6s}  score={r['score']:.4f}  | {r['title'][:50]}")

    # Step 2: Generation
    if not USE_GENERATION:
        print("\n  ⚠️  Generation skipped (USE_GENERATION=False)")
        return

    print("\n🤖 Step 2 — Answer Generation (Gemini 2.5 Flash)")
    try:
        answer = bm25_generate(query, results)
        print("\n  ✅ Answer:")
        print("  " + "-"*60)
        for line in answer.splitlines():
            print(f"  {line}")
        print("  " + "-"*60)
    except Exception as e:
        print(f"\n  ❌ Generation failed: {e}")


if __name__ == "__main__":
    print("\n" + "="*65)
    print("   BM25 RAG — End-to-End Test")
    print("="*65)

    for q in TEST_QUERIES:
        run_test(q)

    print("\n\n🏁 Test complete.")
