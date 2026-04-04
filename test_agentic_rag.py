"""
test_agentic_rag.py
-------------------
End-to-end test for the Agentic RAG pipeline.

Unlike standard RAG or BM25 RAG, the agent:
  • Decides ON ITS OWN which tool(s) to call
  • May search multiple times (e.g. first broad, then domain-specific)
  • Produces a richer, self-consistent answer

Usage:
  python test_agentic_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_rag.agentic_agent import agentic_answer

# ─── Test queries ─────────────────────────────────────────────────────────────
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

VERBOSE = True   # Show the agent's reasoning steps

# ─── Runner ───────────────────────────────────────────────────────────────────
def run(query: str):
    print(f"\n{'='*68}")
    print(f"  QUERY : {query}")
    print(f"{'='*68}")

    result = agentic_answer(query, verbose=VERBOSE)

    print(f"\n  📊 Agent used {result['rounds']} round(s), called {len(result['tools_called'])} tool(s):")
    for call in result["tools_called"]:
        print(f"     ├─ Round {call['round']} | {call['tool']}({call['args']})  → {call['result_summary']}")

    print(f"\n  ✅ Final Answer:")
    print("  " + "─" * 60)
    for line in result["answer"].splitlines():
        print(f"  {line}")
    print("  " + "─" * 60)


if __name__ == "__main__":
    print("\n" + "="*68)
    print("   AGENTIC RAG — End-to-End Test")
    print("="*68)

    for q in TEST_QUERIES:
        run(q)

    print("\n\n🏁 All tests complete.")
