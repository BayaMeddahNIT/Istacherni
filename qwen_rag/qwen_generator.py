"""
qwen_generator.py
-----------------
Generation module for Qwen RAG.
Takes a question + dense-retrieved articles → calls Qwen2.5-7B via Ollama
→ returns a structured Arabic legal answer.
"""

from __future__ import annotations

# ── Make "qwen_rag" importable when run as a plain script ─────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# ── env ────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_MODEL:    str   = os.getenv("OLLAMA_MODEL",      "qwen2.5:7b")
OLLAMA_TIMEOUT:  int   = int(os.getenv("OLLAMA_TIMEOUT",      "120"))
OLLAMA_NUM_CTX:  int   = int(os.getenv("OLLAMA_NUM_CTX",      "4096"))
OLLAMA_TEMP:     float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = """أنت مساعد قانوني متخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين استناداً حصراً إلى المواد القانونية المقدَّمة.

القواعد الصارمة:
1. أجب دائماً باللغة العربية.
2. لا تستخدم أي معلومة خارج نصوص المواد المرفقة.
3. اذكر رقم المادة واسم القانون لكل معلومة تستشهد بها.
4. إذا لم تكفِ المواد للإجابة، قل: "لا تتوفر معلومات كافية في المواد المقدمة."
5. كن دقيقاً، موجزاً، وواضحاً.
6. رتّب إجابتك: ابدأ بالحكم الرئيسي، ثم الشروط، ثم العقوبات إن وُجدت."""


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
    import json as _json
    import urllib.request
    import urllib.error

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = _json.dumps({
        "model":  OLLAMA_MODEL,
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
    user_prompt = (
        f"=== المواد القانونية ذات الصلة ===\n\n{context}\n\n"
        f"=== سؤال المستخدم ===\n\n{question}\n\n=== الإجابة ==="
    )
    return _ollama_chat(system=_SYSTEM, user=user_prompt, max_retries=max_retries)


# ── Health check ───────────────────────────────────────────────────────────────

def check_ollama_health() -> bool:
    import urllib.request
    import json as _json

    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())

        available_models = [m["name"] for m in data.get("models", [])]

        # Accept exact match or ":latest" suffix match
        if (
            OLLAMA_MODEL not in available_models
            and f"{OLLAMA_MODEL}:latest" not in available_models
        ):
            print(
                f"[Qwen-Gen] ⚠  Model '{OLLAMA_MODEL}' not found.\n"
                f"           Available: {available_models}\n"
                f"           Run:  ollama pull {OLLAMA_MODEL}"
            )
            return False

        print(f"[Qwen-Gen] ✓ Ollama healthy | model '{OLLAMA_MODEL}' ready.")
        return True

    except Exception as e:
        print(f"[Qwen-Gen] ✗ Ollama health check failed: {e}")
        return False


if __name__ == "__main__":
    check_ollama_health()
