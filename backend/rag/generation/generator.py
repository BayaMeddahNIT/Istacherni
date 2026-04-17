"""
generator.py
------------
Generation module: takes the user question + retrieved article chunks
and calls Gemini to produce a contextualized Arabic legal answer.

The model is instructed to:
  - Answer ONLY based on the provided articles
  - Cite the article number and law name for each claim
  - Reply in Arabic (matching the dataset language)
  - Politely say it doesn't know if the answer is not in the articles
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from google import genai
from google.genai import types as genai_types

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-2.5-flash"

# Lazy singleton — initialized on first call, not at import time
_genai_client = None
_generation_config = None

def _get_genai_client():
    global _genai_client, _generation_config
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set. Check your .env file.")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
        _generation_config = genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
        )
    return _genai_client

SYSTEM_INSTRUCTION = """أنت مساعد قانوني متخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين بناءً فقط على المواد القانونية المقدمة إليك.

القواعد:
1. أجب دائماً باللغة العربية.
2. استند فقط إلى المواد القانونية المرفقة بسؤال المستخدم.
3. اذكر رقم المادة واسم القانون لكل معلومة تقتبسها.
4. إذا لم تتضمن المواد المقدمة إجابة واضحة، قل: "لا تتوفر لديّ معلومات كافية في المواد المقدمة للإجابة على هذا السؤال."
5. لا تخترع معلومات أو تحكم من خارج النصوص المقدمة.
6. كن دقيقاً ومختصراً وواضحاً."""


def _build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved articles into a readable context block for the prompt."""
    if not retrieved_chunks:
        return "لا توجد مواد قانونية ذات صلة."

    parts = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        law_name = meta.get("law_name", "")
        article_num = meta.get("article_number", "")
        title = meta.get("title", "")
        text = meta.get("text_original", chunk.get("text", ""))

        header = f"【{law_name} — المادة {article_num}】"
        if title:
            header += f" ({title})"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a legal answer using Gemini based on retrieved article chunks.

    Args:
        question:         The user's question (Arabic or French)
        retrieved_chunks: List of dicts from retriever.retrieve()

    Returns:
        A string containing the generated Arabic answer.
    """
    context = _build_context(retrieved_chunks)

    prompt = f"""{SYSTEM_INSTRUCTION}

=== المواد القانونية ذات الصلة ===

{context}

=== سؤال المستخدم ===

{question}

=== الإجابة ==="""

    import re, time as _time
    for attempt in range(5):
        try:
            response = _get_genai_client().models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=_generation_config,
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 4:
                m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err)
                wait = int(m.group(1)) + 3 if m else 30 * (2 ** attempt)
                print(f"[WARN] generation rate-limited, waiting {wait}s...")
                _time.sleep(wait)
            else:
                raise
    raise RuntimeError("generate_answer failed after 5 retries")


if __name__ == "__main__":
    # Quick manual test (requires vector store to exist)

    #from backend.rag.retrieval.retriever import retrieve

    from backend.rag.retrieval.bge_retriever import retrieve

    question = "ما هي شروط المسؤولية المدنية في القانون الجزائري؟"
    print(f"Question: {question}\n")

    #chunks = retrieve(question, top_k=3)
    chunks = retrieve(question, top_k=3)
    
    answer = generate_answer(question, chunks)

    print("Answer:\n")
    print(answer)
