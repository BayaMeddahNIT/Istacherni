"""
minilm_generator.py
-------------------
Generation module for MiniLM hybrid RAG.
Takes a question + hybrid-retrieved articles → calls Gemini 2.5 Flash
→ returns a structured Arabic legal answer.

Completely standalone — uses its own API key (set MINILM_GEMINI_API_KEY in .env,
or falls back to GEMINI_API_KEY if only one key is configured).
"""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# ── Load env ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_API_KEY = os.getenv("MINILM_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-2.5-flash"

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_client = None
_config = None


def _get_client():
    global _client, _config
    if _client is None:
        if not _API_KEY:
            raise EnvironmentError(
                "No API key found. Set MINILM_GEMINI_API_KEY (or GEMINI_API_KEY) in .env"
            )
        _client = genai.Client(api_key=_API_KEY)
        _config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
        )
    return _client, _config


# ── System instruction ─────────────────────────────────────────────────────────
_SYSTEM = """أنت مساعد قانوني متخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين استناداً حصراً إلى المواد القانونية المقدَّمة.

القواعد الصارمة:
1. أجب دائماً باللغة العربية.
2. لا تستخدم أي معلومة خارج نصوص المواد المرفقة.
3. اذكر رقم المادة واسم القانون لكل معلومة تستشهد بها.
4. إذا لم تكفِ المواد للإجابة، قل: "لا تتوفر معلومات كافية في المواد المقدمة."
5. كن دقيقاً، موجزاً، وواضحاً."""


def _build_context(retrieved: list[dict]) -> str:
    """Format hybrid-retrieved articles into a readable context block."""
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

        parts.append(f"{header}\n" + "\n".join(p for p in body_parts if p))

    return "\n\n---\n\n".join(parts)


def minilm_generate(
    question:    str,
    retrieved:   list[dict],
    max_retries: int = 6,
) -> str:
    """
    Generate a legal answer using Gemini based on hybrid-retrieved articles.

    Args:
        question:    The user's question.
        retrieved:   List of dicts from minilm_retrieve().
        max_retries: Number of 429-rate-limit retries before giving up.

    Returns:
        The generated Arabic answer string.
    """
    context = _build_context(retrieved)
    prompt  = f"""{_SYSTEM}

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
            is_last = attempt >= max_retries - 1

            if "429" in err and not is_last:
                # Rate-limited: respect the retryDelay hint or use exponential back-off
                m    = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err)
                wait = int(m.group(1)) + 3 if m else 30 * (2 ** attempt)
                print(f"[MiniLM-Gen] Rate-limited (429), waiting {wait}s… "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

            elif ("503" in err or "UNAVAILABLE" in err) and not is_last:
                # Transient server overload: fixed back-off (15s, 30s, 60s, …)
                wait = 15 * (2 ** attempt)
                print(f"[MiniLM-Gen] Server unavailable (503), waiting {wait}s… "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

            else:
                raise

    raise RuntimeError("minilm_generate: failed after all retries")


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from minilm_rag.minilm_retriever import minilm_retrieve

    question = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Question: {question}\n")

    chunks = minilm_retrieve(question, top_k=5)
    answer = minilm_generate(question, chunks)

    print("Answer:\n")
    print(answer)