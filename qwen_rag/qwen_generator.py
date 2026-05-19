"""
qwen_generator.py
-----------------
Generation module for Qwen RAG.
Takes a question + dense-retrieved articles → calls Qwen2.5-7B via Ollama
→ returns a structured Arabic legal answer.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# ── env ────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL:   str   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")

# UPDATED: Using the real model name available on your machine
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", os.getenv("LOCAL_LLM_MODEL", "qwen2:7b"))

OLLAMA_TIMEOUT:    int   = int(os.getenv("OLLAMA_TIMEOUT",     "600"))
OLLAMA_NUM_CTX:    int   = int(os.getenv("OLLAMA_NUM_CTX",     "2048"))
OLLAMA_TEMP:       float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── Context pollution blacklist ─────────────────────────────────────────────
# Fix A: Generic scope/definitions articles that occupy top-3 context slots
_CONTEXT_BLACKLIST = {
    ("قانون العقوبات",                                        "3"),
    ("قانون الوقاية من الجرائم المتصلة بتكنولوجيات",         "2"),
    ("قانون الوقاية من الجرائم المتصلة بتكنولوجيات",         "3"),
    ("القانون المدني",                                         "1"),
    ("القانون التجاري",                                        "1"),
}

def _is_blacklisted(art: dict) -> bool:
    law = art.get("law_name", "")
    num = str(art.get("article_number", ""))
    return any(
        frag in law and num == num_key
        for frag, num_key in _CONTEXT_BLACKLIST
    )

# ── Arabic output sanitizer ─────────────────────────────────────────────────
# Fix D: Strip any CJK character sequences from model output
_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')

def _sanitize_arabic(text: str) -> str:
    return _CJK_RANGE.sub('[...]‏', text)

# ── Verification Helpers ───────────────────────────────────────────────────

def _verify_citation_alignment(response: str, retrieved: List[Dict[str, Any]]) -> bool:
    """
    Verifies that any quoted legal text in the response is actually present
    verbatim within the retrieved context. Also prevents empty/generic quotes.
    """
    if not retrieved:
        return True

    # Extract Step 1 quote: look for الخطوة 1 or step 1 style outputs
    # Examples: "**الخطوة 1 — ربط المادة:** استخرجت المادة..."
    step1_match = re.search(r'\*\*الخطوة\s*1[^\n:]*:\s*(.*)', response)
    if not step1_match:
        # If the LLM didn't output Step 1 format, it broke rules
        return False

    quote = step1_match.group(1).strip()
    if len(quote) < 8:
        # Quote too short or generic -> invalid
        return False

    # Clean the quote and context of diacritics/whitespace for robust matching
    def clean_text(t: str) -> str:
        # Strip diacritics (harakat)
        t = re.sub(r'[\u064B-\u065F]', '', t)
        # Normalize Alef variants
        t = re.sub(r'[أإآ]', 'ا', t)
        # Normalize Ya / Alef Maqsuora
        t = re.sub(r'[ىي]', 'ي', t)
        # Strip non-alphanumeric/punctuation
        t = re.sub(r'[^\w\s]', '', t)
        return " ".join(t.split())

    cleaned_quote = clean_text(quote)
    if not cleaned_quote:
        return False

    for art in retrieved:
        context_body = clean_text(art.get("text_original", ""))
        # Check if the cleaned quote exists verbatim in the context
        if cleaned_quote in context_body or context_body in cleaned_quote:
            return True

        # Also fallback: if the quote is long, check if the first 4-5 words match
        words = cleaned_quote.split()
        if len(words) >= 4:
            sub_quote = " ".join(words[:4])
            if sub_quote in context_body:
                return True

    return False

# ── System prompt ──────────────────────────────────────────────────────────────
# Roadmap Step 5: Generation Grounding Instruction
_SYSTEM = """أنت مستشار قانوني جزائري دقيق جداً. أجب حصراً بناءً على المواد المرفقة.
عليك الالتزام الصارم بهذه القواعد تحت طائلة الفشل:
1. استند فقط إلى المواد المقدمة. لا تستخدم معلومات خارجية أبداً.
2. ابدأ تحليلك دائماً من المادة [1] باعتبارها الأعلى صلة. اقرأها بتمعن واستخرج الحكم منها إن وجد.
3. اذكر دائماً رقم المادة والقانون المصدر (مثال: المادة 429 من قانون العقوبات).
4. اذكر العقوبة المحددة بالأرقام إن وجدت (سنوات، أشهر، دينار).
5. إذا لم تجد في المواد أي نص صريح، اذكر: 'لا تتضمن المواد المقدمة نصاً على هذه الحالة' — ولكن تأكد مرتين قبل الرفض. الغالبية العظمى من الأسئلة لها إجابة في المادة [1].

اتبع هذا الهيكل الإلزامي (الاستدلال المتسلسل):
**الخطوة 1 — ربط المادة:** استخرج من المادة [1] أو [2] الجملة التي تعالج السؤال بشكل مباشر.
**الخطوة 2 — الحكم القانوني:** صغ الحكم بناءً على الجملة المستخرجة.
**الخطوة 3 — العقوبة/الشروط:** اذكر التفاصيل الرقمية أو الشروط بوضوح."""


# ── Context builder ────────────────────────────────────────────────────────────

def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved articles into a readable Arabic context block."""
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    filtered = [a for a in retrieved if not _is_blacklisted(a)]
    if len(filtered) < len(retrieved):
        removed = len(retrieved) - len(filtered)
        print(f"[Qwen-Gen] Blacklist filtered {removed} generic article(s) from context.")
    retrieved = filtered if filtered else retrieved

    parts = []
    for i, art in enumerate(retrieved):
        header = f"【المادة [{i+1}] - {art.get('law_name', 'قانون غير معروف')} — المادة {art.get('article_number', 'N/A')}】"
        if art.get("title"):
            header += f" ({art['title']})"

        body_lines = []
        original = art.get("text_original", "")
        # Optimize context density: truncate extremely long articles
        if len(original) > 1000:
            original = original[:1000] + " ... [نص مقطوع للطول]"
        body_lines.append(original)

        if art.get("legal_conditions_summary"):
            body_lines.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_lines.append(f"العقوبة: {art['penalties_summary']}")

        parts.append(f"{header}\n" + "\n".join(body_lines))

    return "\n\n---\n\n".join(parts)


# ── Ollama chat call ───────────────────────────────────────────────────────────

def _ollama_chat(
    messages: List[Dict[str, str]],
    max_retries: int = 3,
) -> str:
    import json as _json
    import urllib.request
    import urllib.error

    url     = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = _json.dumps({
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMP,
            "num_ctx":     OLLAMA_NUM_CTX,
        },
        "messages": messages,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            req  = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"].strip()

        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[Qwen-Gen] Ollama unreachable ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
                    "Make sure Ollama is running:  ollama serve"
                ) from e
    return "خطأ في الاتصال بالمولد."


# ── Public API ─────────────────────────────────────────────────────────────────

def qwen_generate(
    question: str,
    retrieved: List[Dict[str, Any]],
    max_retries: int = 3,
) -> str:
    context = _build_context(retrieved)
    user_prompt = f"=== المواد القانونية ذات الصلة ===\n\n{context}\n\n=== سؤال المستخدم ===\n\n{question}\n\n=== الإجابة ==="
    
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    
    attempt = 1
    max_passes = 3
    response = _ollama_chat(messages=messages, max_retries=max_retries)
    
    while attempt < max_passes:
        refusal_keywords = ["لا تتضمن", "لا يوجد نص", "لا أستطيع", "غير مذكور"]
        is_refusal = any(kw in response for kw in refusal_keywords)
        
        is_aligned = _verify_citation_alignment(response, retrieved) if not is_refusal else True
        
        if not is_refusal and is_aligned:
            # Good grounded response!
            break
            
        print(f"[Qwen-Gen] Pass {attempt} failed verification. Refusal: {is_refusal}, Misaligned Citation: {not is_aligned}")
        
        if is_refusal:
            print(f"[Qwen-Gen] Pass {attempt} - Refusal loop triggered.")
            art1 = retrieved[0] if retrieved else {}
            art1_ref = f"{art1.get('law_name', '')} المادة {art1.get('article_number', '')}" if art1 else "المادة [1]"
            corrective_prompt = (
                f"سؤال المستخدم كان: '{question}'. "
                f"أنت ذكرت أنه لا يوجد نص يعالج الحالة، لكن المادة [1] ({art1_ref}) قدمت في السياق. "
                f"أرجو مراجعة المادة [1] بعناية شديدة. هل حقاً لا تحتوي على إجابة أو حكم يخص هذا السؤال بشكل مباشر أو غير مباشر؟ "
                f"إذا وجدت صلة، اكتب الإجابة بناءً عليها فوراً مع الاستدلال والالتزام بالخطوات الثلاث الإلزامية."
            )
        else:
            print(f"[Qwen-Gen] Pass {attempt} - Citation misalignment loop triggered.")
            corrective_prompt = (
                "تنبيه: لقد قمت بالاقتباس بشكل غير صحيح في الخطوة 1 أو لم تلتزم بالاقتباس الحرفي من المواد القانونية المرفقة.\n"
                "أعد كتابة الإجابة مع الالتزام التام بالخطوات الثلاث. "
                "في **الخطوة 1 — ربط المادة:** يجب أن تقتبس جملة موجودة حرفياً وبشكل صريح داخل المادة [1] أو المادة [2] المرفقة."
            )
            
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": corrective_prompt})
        
        response = _ollama_chat(messages=messages, max_retries=max_retries)
        attempt += 1
        
    # Fallback degradation path if verification fails on all 3 passes
    if attempt >= max_passes:
        refusal_keywords = ["لا تتضمن", "لا يوجد نص", "لا أستطيع", "غير مذكور"]
        is_still_refusal = any(kw in response for kw in refusal_keywords)
        is_still_aligned = _verify_citation_alignment(response, retrieved) if not is_still_refusal else True
        
        if not is_still_aligned and not is_still_refusal:
            print(f"[Qwen-Gen] ⚠️ All {max_passes} passes failed verification. Triggering fallback degradation response.")
            art1 = retrieved[0] if retrieved else {}
            art1_ref = f"المادة {art1.get('article_number', '')} من {art1.get('law_name', '')}" if art1 else "المواد المرفقة"
            response = (
                f"**الخطوة 1 — ربط المادة:** استناداً إلى {art1_ref}.\n"
                f"**الخطوة 2 — الحكم القانوني:** يرجى مراجعة نص المادة مباشرة للتأكد من انطباقها الفعلي، حيث لم يتمكن النظام من صياغة استدلال مؤكد متطابق حرفياً مع السؤال الموجه.\n"
                f"**الخطوة 3 — العقوبة/الشروط:** يُنصح بالرجوع إلى المرجعية الرسمية لتجنب أي لبس في التفسير."
            )

    return _sanitize_arabic(response)


# ── Health check ───────────────────────────────────────────────────────────────

def check_ollama_health() -> bool:
    import urllib.request
    import json as _json

    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())
        # Handles both full 'qwen2.5:7b' and short names
        available_models = [m["name"] for m in data.get("models", [])]
        
        # Check for exact match or version-less match
        if OLLAMA_MODEL not in available_models and f"{OLLAMA_MODEL}:latest" not in available_models:
            print(f"[Qwen-Gen] ⚠ Model '{OLLAMA_MODEL}' not found. Available: {available_models}")
            return False
        
        print(f"[Qwen-Gen] ✓ Ollama healthy | model '{OLLAMA_MODEL}' ready.")
        return True
    except Exception as e:
        print(f"[Qwen-Gen] ✗ Ollama health check failed: {e}")
        return False

if __name__ == "__main__":
    check_ollama_health()