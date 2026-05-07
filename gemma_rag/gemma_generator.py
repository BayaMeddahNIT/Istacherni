"""
gemma_generator.py
-----------------
Generation module for Gemma RAG.
Takes a question + retrieved articles → calls Gemma 2 via Ollama
→ returns a structured Arabic legal answer.
"""

from __future__ import annotations

import sys
import os
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# ── Make "gemma_rag" importable when run as a plain script ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
# Use OLLAMA_JAIS_MODEL as specified in your .env for Gemma 2
OLLAMA_GEMMA_MODEL: str = os.getenv("OLLAMA_JAIS_MODEL", "gemma4:4b")
OLLAMA_TIMEOUT:  int   = int(os.getenv("OLLAMA_TIMEOUT",      "180"))
OLLAMA_NUM_CTX:  int   = int(os.getenv("OLLAMA_NUM_CTX",      "1536")) # Lowered for memory stability  2048
OLLAMA_TEMP:     float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """أنت مستشار قانوني خبير ومتخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين استناداً حصراً إلى المواد القانونية المقدَّمة.
لضمان أعلى درجات الدقة، يجب عليك التفكير بعمق وتحليل المواد خطوة بخطوة قبل الإجابة.

القواعد الصارمة:
1. أجب دائماً باللغة العربية بأسلوب قانوني رصين ومفصل.
2. قم بتحليل السؤال بعمق، واربطه بالمواد القانونية المرفقة بشكل منطقية ومفصل.
3. فكر خطوة بخطوة (Chain of Thought): اشرح كيف توصلت إلى الاستنتاج من خلال تفسير كل مادة على حدة.
4. يجب عليك الاقتباس الحرفي لاسم القانون ورقم المادة من المصادر المرفقة عند الإجابة. يمنع منعا باتا اختراع مواد غير موجودة.
5. لا تستخدم أي معلومة خارج نصوص المواد المرفقة، وإذا لم تكفِ المواد للإجابة، قل بوضوح: "لا تتوفر معلومات كافية في المواد المقدمة."
6. رتّب إجابتك: ابدأ بتحليل مفصل للموقف، ثم اذكر الحكم الرئيسي، الشروط، وأخيراً العقوبات إن وُجدت."""

# ── Context builder ────────────────────────────────────────────────────────────

def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved articles into a readable Arabic context block."""
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    parts = []
    for art in retrieved:
        header = (
            f"【{art.get('law_name', 'قانون غير معروف')} "
            f"— المادة {art.get('article_number', 'N/A')}】"
        )
        if art.get("title"):
            header += f" ({art['title']})"

        body_lines = [art.get("text_original", "")]
        if art.get("legal_conditions_summary"):
            body_lines.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_lines.append(f"العقوبة: {art['penalties_summary']}")

        parts.append(f"{header}\n" + "\n".join(body_lines))

    return "\n\n---\n\n".join(parts)


# ── Ollama chat call ───────────────────────────────────────────────────────────

def _ollama_chat(system: str, user: str, max_retries: int = 3) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    print(f"[Gemma-Gen] Calling Ollama model '{OLLAMA_GEMMA_MODEL}' ...")
    payload = json.dumps({
        "model":  OLLAMA_GEMMA_MODEL,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMP,
            "num_ctx":     OLLAMA_NUM_CTX,
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
                print(f"[Gemma-Gen] Ollama unreachable ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
                    f"Make sure Ollama is running and you have pulled '{OLLAMA_GEMMA_MODEL}'"
                ) from e

    return "خطأ في الاتصال بالمولد."


# ── Public API ─────────────────────────────────────────────────────────────────

def gemma_generate(
    question: str,
    retrieved: List[Dict[str, Any]],
    max_retries: int = 3,
) -> str:
    context = _build_context(retrieved)
    user_prompt = (
        f"=== المواد القانونية ذات الصلة ===\n\n{context}\n\n"
        f"=== سؤال المستخدم ===\n\n{question}\n\n=== الإجابة ==="
    )
    return _ollama_chat(system=_SYSTEM, user=user_prompt, max_retries=max_retries)


# ── Health check ───────────────────────────────────────────────────────────────

def check_gemma_health() -> bool:
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())

        available_models = [m["name"] for m in data.get("models", [])]

        if (
            OLLAMA_GEMMA_MODEL not in available_models
            and f"{OLLAMA_GEMMA_MODEL}:latest" not in available_models
        ):
            print(
                f"[Gemma-Gen] ⚠  Model '{OLLAMA_GEMMA_MODEL}' not found.\n"
                f"           Available: {available_models}\n"
                f"           Run:  ollama pull {OLLAMA_GEMMA_MODEL}"
            )
            return False

        print(f"[Gemma-Gen] ✓ Ollama healthy | model '{OLLAMA_GEMMA_MODEL}' ready.")
        return True

    except Exception as e:
        print(f"[Gemma-Gen] ✗ Ollama health check failed: {e}")
        return False


if __name__ == "__main__":
    check_gemma_health()
