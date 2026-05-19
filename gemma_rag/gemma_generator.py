"""
gemma_generator.py
------------------
Shared generation module for all RAG pipelines.

Takes a question + retrieved articles → calls gemma2:9b via Ollama
→ returns a structured Arabic legal answer.

Used by:
  - run_bge_bm25.py      (BGE-M3 + BM25 hybrid retrieval)
  - run_graph_rag.py     (Knowledge Graph retrieval)
  - run_qwen_rag.py      (Qwen dense retrieval)
  - run_camelbert_rag.py (CAMeLBERT dense retrieval)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# ── Env ─────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ── Config ───────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL:    str   = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_GEMMA_MODEL: str   = os.getenv("OLLAMA_GEMMA_MODEL", "gemma2:9b")
OLLAMA_TIMEOUT:     int   = int(os.getenv("OLLAMA_TIMEOUT",     "600"))
OLLAMA_NUM_CTX:     int   = int(os.getenv("OLLAMA_NUM_CTX",     "4096"))
OLLAMA_TEMP:        float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── Context pollution blacklist ─────────────────────────────────────────────
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
_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')

def _sanitize_arabic(text: str) -> str:
    return _CJK_RANGE.sub('[...]‏', text)

# ── System prompt ────────────────────────────────────────────────────────────────
# Roadmap Step 5: Generation Grounding Instruction
_SYSTEM = """مساعد قانوني متخصص في القانون الجزائري.
أجب حصراً بناءً على المواد المرفقة.
قواعد صارمة:
1. ابدأ تحليلك دائماً من المادة [1] باعتبارها الأعلى صلة. لا تتجاهل المادة الأولى لصالح مادة لاحقة.
2. اذكر دائماً رقم المادة واسم القانون لكل معلومة.
3. إذا لم تجد في المواد المقدمة نصاً صريحاً، اذكر: 'لا تتضمن المواد المقدمة نصاً على هذه الحالة'.
4. أجب بنص عادي مباشر وابتعد عن التنسيق المعقد."""


# ── Context builder ──────────────────────────────────────────────────────────────

def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved article dicts into a readable Arabic context block."""
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    filtered = [a for a in retrieved if not _is_blacklisted(a)]
    if len(filtered) < len(retrieved):
        removed = len(retrieved) - len(filtered)
        print(f"[Gemma-Gen] Blacklist filtered {removed} generic article(s) from context.", flush=True)
    retrieved = filtered if filtered else retrieved

    parts = []
    for art in retrieved:
        law  = art.get("law_name",       "قانون غير معروف")
        num  = art.get("article_number", "N/A")
        header = f"【{law} — المادة {num}】"
        if art.get("title"):
            header += f" ({art['title']})"

        body_lines = [art.get("text_original", "")]
        if art.get("legal_conditions_summary"):
            body_lines.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_lines.append(f"العقوبة: {art['penalties_summary']}")

        parts.append(f"{header}\n" + "\n".join(body_lines))

    return "\n\n---\n\n".join(parts)


# ── Ollama call ──────────────────────────────────────────────────────────────────

def _ollama_chat(system: str, user: str, max_retries: int = 3) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model":  OLLAMA_GEMMA_MODEL,
        "stream": False,
        "options": {
            "temperature":    OLLAMA_TEMP,
            "num_ctx":        OLLAMA_NUM_CTX,
            "num_predict":    800,    # Hard token cap — prevents runaway long answers (~2× speedup)
            "top_k":          20,     # Narrow sampling = fewer token evaluations
            "top_p":          0.8,
            "repeat_penalty": 1.1,   # Prevent repetition loops that inflate length
            "stop":           ["\n\n\n", "السؤال:"],
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"].strip()

        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[Gemma-Gen] Ollama unreachable ({e}), retrying in {wait}s …", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
                    "Make sure Ollama is running:  ollama serve"
                ) from e

    return "خطأ في الاتصال بالمولد."


# ── Public API ───────────────────────────────────────────────────────────────────

def gemma_generate(
    question:   str,
    retrieved:  List[Dict[str, Any]],
    max_retries: int = 3,
) -> str:
    """
    Generate a legal answer using gemma2:9b via Ollama.

    Args:
        question:    The user's Arabic legal question.
        retrieved:   List of article dicts from any retriever.
        max_retries: Number of Ollama connection retry attempts.

    Returns:
        Generated Arabic answer string.
    """
    context = _build_context(retrieved)
    user_prompt = (
        "=== المواد القانونية ذات الصلة ===\n\n"
        f"{context}\n\n"
        "=== سؤال المستخدم ===\n\n"
        f"{question}\n\n"
        "=== الإجابة ==="
    )
    response = _ollama_chat(system=_SYSTEM, user=user_prompt, max_retries=max_retries)
    return _sanitize_arabic(response)


# ── Health check ─────────────────────────────────────────────────────────────────

def check_ollama_health() -> bool:
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        available = [m["name"] for m in data.get("models", [])]
        model_ok = (
            OLLAMA_GEMMA_MODEL in available
            or f"{OLLAMA_GEMMA_MODEL}:latest" in available
        )
        if not model_ok:
            print(f"[Gemma-Gen] ⚠ '{OLLAMA_GEMMA_MODEL}' not found. Available: {available}")
            return False
        print(f"[Gemma-Gen] ✓ Ollama healthy | model '{OLLAMA_GEMMA_MODEL}' ready.")
        return True
    except Exception as e:
        print(f"[Gemma-Gen] ✗ Health check failed: {e}")
        return False


if __name__ == "__main__":
    check_ollama_health()
