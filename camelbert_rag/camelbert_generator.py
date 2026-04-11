"""
camelbert_generator.py
----------------------
Generation module for the CAMeLBERT dense-vector RAG.

Takes a question + CAMeLBERT-retrieved articles → calls Gemini 2.5 Flash
→ returns a structured Arabic legal answer.

The *retrieval* step is now fully local (CAMeLBERT + FAISS); only the
*generation* step still calls the Gemini API, identical in behaviour to
bm25_generator.py so the two pipelines are easy to compare side-by-side.

API key precedence:
  1. CAMELBERT_GEMINI_API_KEY  (dedicated key for this pipeline)
  2. GEMINI_API_KEY            (shared fallback)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# ── Load env ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_API_KEY = os.getenv("CAMELBERT_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-2.5-flash"

# ── Lazy singleton ──────────────────────────────────────────────────────────────
_client = None
_config = None


def _get_client():
    global _client, _config
    if _client is None:
        if not _API_KEY:
            raise EnvironmentError(
                "No API key found. Set CAMELBERT_GEMINI_API_KEY (or GEMINI_API_KEY) in .env"
            )
        _client = genai.Client(api_key=_API_KEY)
        _config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
        )
    return _client, _config


# ── System instruction (identical to BM25 pipeline for fair comparison) ─────────
_SYSTEM = """أنت مساعد قانوني متخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين استناداً حصراً إلى المواد القانونية المقدَّمة.

القواعد الصارمة:
1. أجب دائماً باللغة العربية.
2. لا تستخدم أي معلومة خارج نصوص المواد المرفقة.
3. اذكر رقم المادة واسم القانون لكل معلومة تستشهد بها.
4. إذا لم تكفِ المواد للإجابة، قل: "لا تتوفر معلومات كافية في المواد المقدمة."
5. كن دقيقاً، موجزاً، وواضحاً."""


def _build_context(retrieved: list[dict]) -> str:
    """Format retrieved CAMeLBERT articles into a readable context block."""
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    parts = []
    for art in retrieved:
        header = f"【{art['law_name']} — المادة {art['article_number']}】"
        if art.get("title"):
            header += f" ({art['title']})"
        body_parts = [art.get("text_original", "")]
        if art.get("legal_conditions_summary"):
            body_parts.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_parts.append(f"العقوبة: {art['penalties_summary']}")
        parts.append(f"{header}\n" + "\n".join(body_parts))

    return "\n\n---\n\n".join(parts)


def camelbert_generate(
    question: str,
    retrieved: list[dict],
    max_retries: int = 4,
) -> str:
    """
    Generate a legal answer using Gemini based on CAMeLBERT-retrieved articles.

    Args:
        question:    The user's question (Arabic).
        retrieved:   List of dicts from camelbert_retrieve().
        max_retries: Number of 429-retry attempts before raising.

    Returns:
        The generated Arabic answer string.
    """
    context = _build_context(retrieved)
    prompt = f"""{_SYSTEM}

=== المواد القانونية ذات الصلة ===

{context}

=== سؤال المستخدم ===

{question}

=== الإجابة ==="""

    client, config = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=config,
            )
            return response.text.strip()

        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err)
                wait = int(m.group(1)) + 3 if m else 30 * (2 ** attempt)
                print(f"[CameLBERT-Gen] Rate-limited, waiting {wait}s …")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("camelbert_generate: failed after all retries")


# ── CLI smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from camelbert_rag.camelbert_retriever import camelbert_retrieve

    q = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Question: {q}\n")
    chunks = camelbert_retrieve(q, top_k=5)
    answer = camelbert_generate(q, chunks)
    print("Answer:\n")
    print(answer)