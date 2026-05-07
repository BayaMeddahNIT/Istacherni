"""
jais_generator.py
-----------------
Generation module for Jais RAG (LOCAL VERSION).
Takes a question + dense-retrieved articles → calls Jais via local Ollama.
→ returns a structured Arabic legal answer.

This version runs entirely on your machine via Ollama.
No API keys or internet connection required for generation.
"""

from __future__ import annotations

# ── Make "jais_rag" importable when run as a plain script ─────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any

from dotenv import load_dotenv

# ── env ────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ── Config (Local Ollama) ──────────────────────────────────────────────────────
OLLAMA_BASE_URL: str   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
# Recommended: jwnder/jais-adaptive:7b or jwnder/jais-adaptive:13b
OLLAMA_JAIS_MODEL: str = os.getenv("OLLAMA_JAIS_MODEL", "gemma4:4b")
OLLAMA_TIMEOUT:  int   = int(os.getenv("OLLAMA_TIMEOUT",      "180"))
OLLAMA_NUM_CTX:  int   = int(os.getenv("OLLAMA_NUM_CTX",      "4096"))
OLLAMA_TEMP:     float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── System prompt (Optimized for Arabic Legal Advisory) ────────────────────────
_SYSTEM = """أنت مستشار قانوني خبير ومتخصص في القانون الجزائري.
أجب باختصار ودقة استناداً للمواد المرفقة.
القواعد:
1. أسلوب قانوني رصين ومختصر.
2. حلل السؤال واربطه بالمواد بشكل منطقي (بدون إطالة غير ضرورية).
3. فكر خطوة بخطوة (Chain of Thought) لكن بتركيز عالي.
4. يجب عليك الاقتباس الحرفي لاسم القانون ورقم المادة من المصادر المرفقة عند الإجابة. يمنع منعا باتا اختراع مواد غير موجودة.
5. رتّب إجابتك: (التحليل، الحكم، الشروط، العقوبة)."""


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

        # Truncate to save context space
        body_text = art.get("text_original", "")[:700]
        body_lines = [body_text]
        
        if art.get("legal_conditions_summary"):
            body_lines.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_lines.append(f"العقوبة: {art['penalties_summary']}")

        parts.append(f"{header}\n" + "\n".join(body_lines))

    return "\n\n---\n\n".join(parts)


# ── Ollama chat call ───────────────────────────────────────────────────────────

def _ollama_chat(system: str, user: str, max_retries: int = 3) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    
    payload = json.dumps({
        "model":  OLLAMA_JAIS_MODEL,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMP,
            "num_ctx":     OLLAMA_NUM_CTX,
            "num_predict": 350,
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
            
            content = body.get("message", {}).get("content", "").strip()
            
            if content:
                print(f"[Gemma-Gen] Response received ({len(content)} chars)")
            else:
                print(f"[Gemma-Gen] ⚠ WARNING: Received empty response from model.")
                
            return content

        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[Jais-Local] Ollama unreachable ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
                    f"Make sure Ollama is running and you have run: ollama pull {OLLAMA_JAIS_MODEL}"
                ) from e

    return "خطأ في الاتصال بالمولد المحلي."


# ── Public API ─────────────────────────────────────────────────────────────────

def jais_generate(
    question: str,
    retrieved: List[Dict[str, Any]],
    max_retries: int = 3,
) -> str:
    """
    Main entry point — matches the signature of qwen_generate.
    """
    context = _build_context(retrieved)
    user_prompt = (
        f"=== المواد القانونية ذات الصلة ===\n\n{context}\n\n"
        f"=== سؤال المستخدم ===\n\n{question}\n\n=== الإجابة ==="
    )
    return _ollama_chat(system=_SYSTEM, user=user_prompt, max_retries=max_retries)


# ── Health check ───────────────────────────────────────────────────────────────

def check_jais_health() -> bool:
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())

        available_models = [m["name"] for m in data.get("models", [])]

        if OLLAMA_JAIS_MODEL not in available_models and f"{OLLAMA_JAIS_MODEL}:latest" not in available_models:
            print(f"[Jais-Local] ⚠ Model '{OLLAMA_JAIS_MODEL}' not found in Ollama.")
            print(f"             Run: ollama pull {OLLAMA_JAIS_MODEL}")
            return False

        print(f"[Jais-Local] ✓ Ollama healthy | model '{OLLAMA_JAIS_MODEL}' ready.")
        return True

    except Exception as e:
        print(f"[Jais-Local] ✗ Ollama health check failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL QWEN GENERATOR CODE (Preserved as comments)
# ─────────────────────────────────────────────────────────────────────────────
# from qwen_rag.qwen_generator import qwen_generate as _qwen_orig
#
# def qwen_generate_backup(question, retrieved):
#     # This points to your original Qwen logic if you ever need to switch back
#     return _qwen_orig(question, retrieved)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    check_jais_health()
