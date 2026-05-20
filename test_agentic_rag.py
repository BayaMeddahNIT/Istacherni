"""
test_agentic_rag.py
-------------------
End-to-end test for the Hybrid Agentic RAG pipeline.

The agent routes queries automatically:
  - Substantive legal questions  → graph_retrieve (local vector graph)
  - Procedural/admin questions   → web_search (DuckDuckGo → official Algerian domains)
"""

import sys
import json
import os
from pathlib import Path

# Evaluation Settings
EVAL_FILE = "evaluation/generated_answers.txt"
os.makedirs("evaluation", exist_ok=True)

def save_result(record: dict, filepath: str):
    """Appends a human-readable evaluation record to a .txt file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"QUERY: {record['query']}\n")
        f.write(f"RETRIEVED ARTICLES: {', '.join(record['retrieved_articles'])}\n")
        f.write(f"USED FALLBACK: {record['used_fallback']}\n")
        f.write(f"ANSWER:\n{record['answer']}\n")
        f.write(f"{'='*60}\n\n")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_rag.agentic_agent import agentic_answer

# ── Canary Queries ─────────────────────────────────────────────────────────────
# 2 SUBSTANTIVE (must route to graph_retrieve)
# 2 PROCEDURAL  (must route to web_search)
TEST_QUERIES = [
    # ("SUBSTANTIVE", "ما هي شروط صحة العقد؟"),        # → Civil Code Arts. 59, 94, 96
    # ("SUBSTANTIVE", "ما الفرق بين SARL و SPA؟"),      # → Commercial Code Arts. 564, 592
    # ("PROCEDURAL",  "كيف أستخرج سجل تجاري لشخص طبيعي؟"), # → PR_CNRC_01 (Offline)
    # ("PROCEDURAL",  "كيف أتحصل على رخصة تجارية؟"),    # → PR_CNRC_03 (Offline)
    # ("SUBSTANTIVE", "ما هي أركان جريمة خيانة الأمانة في السياق التجاري؟"),
    # ("PROCEDURAL",  "كيف أستخرج شهادة حسن السيرة والسلوك؟"),
    #("SUBSTANTIVE", "هل يمكن تعديل عقد تجاري بعد توقيعه؟"),
    #("PROCEDURAL",  "ما هي إجراءات الحصول على شهادة الميلاد رقم 12؟")
    # ── Complex Reasoning Questions (Graph Logic) ─────────────────────────────
    # These require graph traversal and synthesis, testing PPR
    #("SUBSTANTIVE", "ما هي العقوبات المترتبة على الغش التجاري في قانون العقوبات الجزائري؟"),
    #("SUBSTANTIVE", "إذا باعني شخص منتجاً مقلداً، هل يعتبر هذا احتيالاً؟"),
    #("SUBSTANTIVE", "ما هي المسؤولية القانونية لصاحب العمل في حالة إصابة عامل أثناء العمل؟"),
    
    # ── Procedural Questions (Web Search) ──────────────────────────────────────
    # These should route to web_search
    ("PROCEDURAL",  "كيف أستخرج شهادة الميلاد رقم 12؟"),
    #("PROCEDURAL",  "ما هي الوثائق المطلوبة لتسجيل شركة في الجزائر؟"),
    #("PROCEDURAL",  "كيف أتقدم بشكوى ضد موظف إداري؟")
    # ── Boundary Cases (Testing robustness) ──────────────────────────────────────
    # These test slang, abbreviations, and ambiguous references
    #("SUBSTANTIVE", "شيك بلا رصيد واش يدير؟"),          # Slang for "شيك بدون رصيد"
    #("PROCEDURAL",  "سجل تجاري جديد لوحدة جوارية؟"), # Vague reference
    #("SUBSTANTIVE", "بيع سلع مغشوشة قانونا؟")        # Legal jargon
    # ── Domain Integration (Cross-Domain Reasoning) ────────────────────────────
    # Tests seamless navigation between domains via shared entities
    #("SUBSTANTIVE", "ما هي العقوبات المترتبة على الغش التجاري؟"), # Commercial → Penal
    #("SUBSTANTIVE", "إذا تأسست شركة ثم أفلست، ما هي الإجراءات القانونية؟"), # Commercial → Civil/Commercial
    ("SUBSTANTIVE", "هل ي   مكنني توظيف أجنبي في شركتي؟"), # Commercial → Labor/Administrative
    # ── Legal Concepts (Testing specific legal principles) ──────────────────────
    # These test understanding of legal concepts rather than simple facts
    #("SUBSTANTIVE", "ما هي المسؤولية التقصيرية في القانون المدني؟"), # TORT liability
    #("SUBSTANTIVE", "ما الفرق بين البطلان المطلق والبطلان النسبي؟"), # Nullity concepts
    #("SUBSTANTIVE", "ما هي أركان جريمة خيانة الأمانة؟"), # Crime elements
    ("SUBSTANTIVE", "ما هي حقوق العامل في حالة الفصل التعسفي؟"), # Labor rights
]

VERBOSE = True


def run(expected_intent: str, query: str):
    print(f"\n{'='*68}")
    print(f"  [{expected_intent}] QUERY : {query}")
    print(f"{'='*68}")

    result = agentic_answer(query, verbose=VERBOSE)

    # Logging + Saving Layer for Evaluation
    save_result({
        "query": query,
        "retrieved_articles": result.get("retrieved_ids", []),
        "answer": result.get("answer", ""),
        "used_fallback": not result.get("is_context_sufficient", True)
    }, EVAL_FILE)

    tools = result["tools_called"]
    print(f"\n  📊 Agent: {result['rounds']} round(s), {len(tools)} tool call(s):")
    for i, call in enumerate(tools):
        step_num = call.get("round", i + 1)
        tool_name = call.get("tool", "state_action")
        args = call.get("args", {})
        summary = call.get("result_summary", "Completed")
        print(f"     ├─ Step {step_num} | {tool_name}({args})  → {summary}")

    print(f"\n  ✅ Final Answer:")
    print("  " + "─" * 60)
    for line in result["answer"].splitlines():
        print(f"  {line}")
    print("  " + "─" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 68)
    print("   HYBRID AGENTIC RAG — Canary Test Suite")
    print("   Brain: qwen2:7b (Ollama)  |  Graph RAG + DuckDuckGo")
    print("=" * 68)

    for expected_intent, q in TEST_QUERIES:
        run(expected_intent, q)

    print("\n\n🏁 All canary tests complete.")
