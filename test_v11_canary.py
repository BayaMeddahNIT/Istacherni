"""
test_v11_canary.py
------------------
Canary test for v1.1 upgrades:
  1. O(1) CHITCHAT interceptor (Arabic, English, French)
  2. Multilingual router JSON schema (detected_language field)
  3. Dynamic _build_qa_system language directive
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTHONIOENCODING"] = "utf-8"

from agentic_rag.agentic_agent import (
    _check_instant_chitchat,
    _CHITCHAT_RESPONSES,
    _detect_language_fast,
    _build_qa_system,
    _run_router,
)

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"

results = []

def check(name, condition, actual=None):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if not condition and actual is not None:
        print(f"      [DEBUG] Actual: {actual}")
    results.append(condition)

print("\n" + "="*60)
print("  v1.1 Canary Test")
print("="*60)

# ── 1. Language detection ─────────────────────────────────────
print("\n[1] Language Detection (O(n), no LLM)")
check("Arabic script → AR", _detect_language_fast("مرحبا") == "AR")
check("French markers → FR", _detect_language_fast("Quelles sont les conditions") == "FR")
check("English default → EN", _detect_language_fast("What is the penalty") == "EN")

# ── 2. O(1) CHITCHAT interceptor ─────────────────────────────
print("\n[2] Instant CHITCHAT Interceptor (zero LLM)")
is_c, lang = _check_instant_chitchat("مرحبا")
check("'مرحبا' detected as CHITCHAT", is_c)
check("'مرحبا' language is AR", lang == "AR")

is_c, lang = _check_instant_chitchat("hello")
check("'hello' detected as CHITCHAT", is_c)
check("'hello' language is EN", lang == "EN")

is_c, lang = _check_instant_chitchat("bonjour")
check("'bonjour' detected as CHITCHAT", is_c)
check("'bonjour' language is FR", lang == "FR")

is_c, _ = _check_instant_chitchat("ما هي عقوبة السرقة؟")
check("Legal question NOT intercepted", not is_c)

is_c, _ = _check_instant_chitchat("What is the difference between SARL and SPA?")
check("English legal question NOT intercepted", not is_c)

# ── 3. CHITCHAT responses exist for all languages ─────────────
print("\n[3] CHITCHAT Responses")
check("AR response exists", bool(_CHITCHAT_RESPONSES.get("AR")))
check("EN response exists", bool(_CHITCHAT_RESPONSES.get("EN")))
check("FR response exists", bool(_CHITCHAT_RESPONSES.get("FR")))

# ── 4. Dynamic QA system prompt ───────────────────────────────
print("\n[4] Dynamic QA System Builder")
ar_prompt = _build_qa_system("AR")
en_prompt = _build_qa_system("EN")
fr_prompt = _build_qa_system("FR")
check("AR prompt has Arabic directive", "العربية الفصحى" in ar_prompt)
check("EN prompt has English directive", "fluent, professional English" in en_prompt)
check("FR prompt has French directive", "français fluide" in fr_prompt)
check("All prompts have is_context_sufficient", all("is_context_sufficient" in p for p in [ar_prompt, en_prompt, fr_prompt]))
check("All prompts have focused sub-point rule", all("SPECIFIC sub-point" in p for p in [ar_prompt, en_prompt, fr_prompt]))

# ── 5. Router LLM JSON schema (live inference) ───────────────
print("\n[5] Router LLM — New JSON Schema (live inference, ~10s each)")
print("    Testing Arabic legal query...")
r = _run_router("ما هي عقوبة السرقة في القانون الجزائري؟", [], verbose=False)
check("Router returns detected_language", "detected_language" in r)
check("Arabic query → AR language", r.get("detected_language") == "AR")
check("Arabic query → SUBSTANTIVE intent", r.get("intent") == "SUBSTANTIVE")
check("Arabic query has rewritten_query", bool(r.get("rewritten_query")))

print("\n    Testing English query...")
r_en = _run_router("What is the penalty for theft in Algerian law?", [], verbose=False)
check("English query → EN language", r_en.get("detected_language") == "EN", r_en)
rewritten_en = r_en.get("rewritten_query") or ""
check("English query has Arabic rewritten_query", any('\u0600' <= c <= '\u06FF' for c in rewritten_en), rewritten_en)

print("\n    Testing CHITCHAT via router...")
r_chat = _run_router("مرحبا كيف حالك", [], verbose=False)
check("CHITCHAT intent detected", r_chat.get("intent") == "CHITCHAT", r_chat)
check("CHITCHAT has null rewritten_query", r_chat.get("rewritten_query") is None, r_chat.get("rewritten_query"))

print("\n    Testing Chinese leakage prevention...")
r_proc = _run_router("كيف أستخرج سجل تجاري؟", [], verbose=False)
rewritten = r_proc.get("rewritten_query") or ""
check("No Chinese characters in rewritten query", not any('\u4e00' <= c <= '\u9fff' for c in rewritten), rewritten)
check("Procedural query is in Arabic", any('\u0600' <= c <= '\u06FF' for c in rewritten), r_proc)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(results)
total = len(results)
print(f"  Result: {passed}/{total} tests passed")
if passed == total:
    print("  🏆 All checks GREEN — v1.1 is production-ready!")
else:
    print(f"  ⚠️  {total - passed} checks failed — review output above.")
print("="*60 + "\n")
