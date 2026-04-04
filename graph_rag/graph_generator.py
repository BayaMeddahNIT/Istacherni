"""
graph_generator.py
------------------
Generation module for Graph RAG.
Takes a question + graph-retrieved articles → Gemini 2.5 Flash → Arabic answer.

Uses GRAPH_GEMINI_API_KEY if set in .env, falls back to GEMINI_API_KEY.
"""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_API_KEY = os.getenv("GRAPH_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL    = "gemini-2.5-flash"

_client = None
_config = None

def _get_client():
    global _client, _config
    if _client is None:
        if not _API_KEY:
            raise EnvironmentError(
                "No API key found. Set GRAPH_GEMINI_API_KEY or GEMINI_API_KEY in .env"
            )
        _client = genai.Client(api_key=_API_KEY)
        _config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
        )
    return _client, _config


_SYSTEM = """أنت مساعد قانوني متخصص في القانون الجزائري.
تم استرجاع المواد القانونية التالية باستخدام بحث مبني على الرسم البياني المعرفي (Knowledge Graph).

قواعد الإجابة:
1. أجب دائماً باللغة العربية.
2. استند حصرياً إلى المواد الواردة أدناه.
3. اذكر رقم المادة واسم القانون لكل معلومة.
4. إن لم تكفِ المواد، قل ذلك صراحةً.
5. لا تخترع أي معلومة خارج النصوص المقدمة."""


def _build_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."
    parts = []
    for art in retrieved:
        header = f"【{art['law_name']} — المادة {art['article_number']}】"
        if art.get("title"):
            header += f"  ({art['title']})"
        body = [art.get("text_original", "")]
        if art.get("legal_conditions_summary"):
            body.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body.append(f"العقوبة: {art['penalties_summary']}")
        if art.get("keywords"):
            body.append(f"الكلمات المفتاحية: {', '.join(art['keywords'])}")
        # Append graph score info so the LLM knows these were retrieved via graph
        body.append(f"[graph_score={art.get('graph_score', 0):.3f}  pagerank={art.get('pagerank', 0):.6f}]")
        parts.append(f"{header}\n" + "\n".join(body))
    return "\n\n---\n\n".join(parts)


def graph_generate(question: str, retrieved: list[dict], max_retries: int = 4) -> str:
    """
    Generate a legal answer using Gemini based on graph-retrieved articles.

    Args:
        question:   The user's legal question.
        retrieved:  List of dicts from graph_retrieve().
        max_retries: 429-retry limit.

    Returns:
        Arabic answer string.
    """
    context = _build_context(retrieved)
    prompt = f"""{_SYSTEM}

=== المواد القانونية المسترجعة عبر الرسم البياني المعرفي ===

{context}

=== سؤال المستخدم ===

{question}

=== الإجابة ==="""

    client, config = _get_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err)
                wait = int(m.group(1)) + 3 if m else 30 * (2 ** attempt)
                print(f"[GraphGen] Rate-limited, waiting {wait}s…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("graph_generate: failed after all retries")


if __name__ == "__main__":
    from graph_rag.graph_retriever import graph_retrieve
    q = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Question: {q}\n")
    chunks = graph_retrieve(q, top_k=5)
    answer = graph_generate(q, chunks)
    print("Answer:\n", answer)
