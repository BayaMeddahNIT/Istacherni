"""
hybrid_bge_qwen_generator.py
-----------------------------
Generation module: takes retrieved article chunks + user question
and calls Qwen2.5:7b (running via Ollama locally) to produce
a contextualized Arabic legal answer.

The model is instructed to:
  - Answer ONLY based on the provided articles
  - Cite the article number and law name for each claim
  - Reply in Arabic (matching the dataset language)
  - Politely say it doesn't know if the answer is not in the articles

Requirements:
  - Ollama must be running: `ollama serve`
  - Qwen2.5:7b must be pulled: `ollama pull qwen2.5:7b`
  - OLLAMA_BASE_URL defaults to http://localhost:11434

Usage:
  from bm25_rag.hybrid_bge_qwen_generator import qwen_generate
  answer = qwen_generate(question, retrieved_chunks)

-------------------------------------------------------------------
NOTE: The old Gemini-based generator (bm25_generator.py) is kept
as-is and can still be imported independently. The code below is
the NEW Qwen2.5:7b-backed generator.
-------------------------------------------------------------------

# ── OLD GEMINI GENERATOR REFERENCE (kept as comment) ──────────────
#
# from bm25_rag.bm25_generator import bm25_generate
#
# OLD call:
#   from google import genai
#   from google.genai import types as genai_types
#   _API_KEY = os.getenv("BM25_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
#   GENERATION_MODEL = "gemini-2.5-flash"
#   _client = genai.Client(api_key=_API_KEY)
#   _config = genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=1024)
#   response = _client.models.generate_content(
#       model=GENERATION_MODEL,
#       contents=prompt,
#       config=_config,
#   )
#   return response.text.strip()
# ──────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Generation parameters — keep temperature low for factual legal answers
TEMPERATURE  = 0.1
NUM_PREDICT  = 2048   # max tokens to generate


# ── System instruction (Arabic legal assistant) ────────────────────────────────
_SYSTEM = """أنت مساعد قانوني متخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين استناداً حصراً إلى المواد القانونية المقدَّمة.

القواعد الصارمة:
1. أجب دائماً باللغة العربية.
2. لا تستخدم أي معلومة خارج نصوص المواد المرفقة.
3. اذكر رقم المادة واسم القانون لكل معلومة تستشهد بها.
4. إذا لم تكفِ المواد للإجابة، قل: "لا تتوفر معلومات كافية في المواد المقدمة."
5. كن دقيقاً، موجزاً، وواضحاً."""


# ── Context builder ────────────────────────────────────────────────────────────

def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved BM25 + BGE-M3 articles into a readable context block."""
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    parts = []
    for art in retrieved:
        header = f"【{art.get('law_name', '')} — المادة {art.get('article_number', '')}】"
        if art.get("title"):
            header += f" ({art['title']})"

        body_parts = [art.get("text_original", "")]
        if art.get("legal_conditions_summary"):
            body_parts.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_parts.append(f"العقوبة: {art['penalties_summary']}")

        parts.append(f"{header}\n" + "\n".join(p for p in body_parts if p))

    return "\n\n---\n\n".join(parts)


# ── Ollama call ────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    """
    Call Ollama's /api/generate endpoint with Qwen2.5:7b.

    Returns the generated text string.
    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    url     = f"{OLLAMA_BASE_URL}/api/generate"
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"[QwenGen] Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`).\n"
            f"Original error: {e}"
        ) from e


# ── Public API ─────────────────────────────────────────────────────────────────

def qwen_generate(
    question:  str,
    retrieved: List[Dict[str, Any]],
) -> str:
    """
    Generate a legal answer using Qwen2.5:7b (via Ollama) based on
    hybrid BM25 + BGE-M3 retrieved articles.

    Args:
        question:   The user's legal question (Arabic / French / English).
        retrieved:  List of article dicts from hybrid_retrieve().

    Returns:
        Generated Arabic answer string.

    Raises:
        RuntimeError: If Ollama is unreachable.
    """
    context = _build_context(retrieved)

    prompt = f"""{_SYSTEM}

=== المواد القانونية ذات الصلة ===

{context}

=== سؤال المستخدم ===

{question}

=== الإجابة ==="""

    return _call_ollama(prompt)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bm25_rag.hybrid_bge_qwen_retriever import hybrid_retrieve

    question = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Question: {question}\n")
    print(f"[INFO] Using model : {OLLAMA_MODEL}")
    print(f"[INFO] Ollama URL  : {OLLAMA_BASE_URL}\n")

    print("[1/2] Retrieving articles (BM25 + BGE-M3)…")
    chunks = hybrid_retrieve(question, top_k=5)
    print(f"      Found {len(chunks)} articles.\n")

    print("[2/2] Generating answer with Qwen2.5:7b…")
    answer = qwen_generate(question, chunks)

    print("\n" + "═" * 65)
    print("ANSWER:")
    print("═" * 65)
    print(answer)
