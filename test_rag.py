"""
test_rag.py
-----------
End-to-end test of the full Standard RAG pipeline.

Steps:
  1. Retrieve top-K articles for each test query
  2. Generate Arabic answers using Gemini

Usage:
  python test_rag.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
USE_GENERATION = True   # 🔁 ضع False إذا كنت تريد اختبار retrieval فقط
SLEEP_BEFORE_GEN = 3    # ⏱️ حماية من rate limit

# ─── 15 Test Questions (from questions.txt — all 5 legal domains) ─────────────
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

from backend.rag.retrieval.retriever import retrieve

if USE_GENERATION:
    from backend.rag.generation.generator import generate_answer


# ─── Helpers ──────────────────────────────────────────────────────────────────
def safe_generate(query, chunks, max_retries=5):
    """Generation with retry + exponential backoff"""
    for attempt in range(max_retries):
        try:
            print(f"  ⏳ Attempt {attempt+1}...")
            time.sleep(SLEEP_BEFORE_GEN)
            response = generate_answer(query, chunks)
            return response
        except Exception as e:
            wait_time = min(2 ** attempt, 60)
            print(f"  ⚠️ Generation failed: {e}")
            print(f"  ⏳ Retrying in {wait_time}s...\n")
            time.sleep(wait_time)
    return None


# ─── Runner ───────────────────────────────────────────────────────────────────
def run_test(query: str):
    print(f"\n{'='*65}")
    print(f"  QUERY: {query}")
    print(f"{'='*65}")

    # Step 1: Retrieval
    print("\n🔎 Step 1 — Retrieval")
    try:
        chunks = retrieve(query, top_k=3)
        print(f"  Retrieved {len(chunks)} articles:")
        for c in chunks:
            m = c["metadata"]
            print(f"    - {m.get('law_name','')} المادة {m.get('article_number','')} | score={c['score']:.4f}")
    except Exception as e:
        print(f"  ❌ Retrieval failed: {e}")
        return

    # Step 2: Generation
    if not USE_GENERATION:
        print("\n  ⚠️ Generation skipped (USE_GENERATION=False)")
        return

    print("\n🤖 Step 2 — Answer Generation")
    answer = safe_generate(query, chunks)

    if answer:
        print("\n  ✅ Answer:")
        print("  " + "-" * 60)
        for line in answer[:800].splitlines():
            print(f"  {line}")
        print("  " + "-" * 60)
    else:
        print("  ❌ Generation failed after multiple attempts.")
        print("\n  📚 Fallback: Retrieved context:\n")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['text'][:300]}...\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("   STANDARD RAG — End-to-End Test")
    print("="*65)

    for q in TEST_QUERIES:
        run_test(q)

    print("\n\n🏁 Test completed.")