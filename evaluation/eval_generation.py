"""
eval_generation.py
------------------
LLM-as-a-Judge: uses Gemini to score each RAG model's generated answer
on three dimensions:

  • Faithfulness  (1–5): Only uses facts from the retrieved articles?
  • Relevance     (1–5): Actually addresses the question?
  • Completeness  (1–5): Thorough, not just a partial response?

Uses EVAL_GEMINI_API_KEY (dedicated judge key with full quota).
NEVER uses GEMINI_API_KEY — that key is reserved exclusively for Standard RAG.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from evaluation.eval_testset import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# EVAL_GEMINI_API_KEY is the dedicated judge key (reuses AGENTIC key which has
# full quota and is called sequentially — never simultaneously with the judge).
# We intentionally DO NOT fall back to GEMINI_API_KEY to protect Standard RAG's quota.
_API_KEY = (
    os.getenv("EVAL_GEMINI_API_KEY")
    or os.getenv("AGENTIC_GEMINI_API_KEY")  # fallback if EVAL key not set
    or os.getenv("BM25_GEMINI_API_KEY")     # secondary fallback
    # NOTE: GEMINI_API_KEY intentionally omitted — reserved for Standard RAG only
)
JUDGE_MODEL = "gemini-2.0-flash"   # stable, 1,500 free req/day
GEN_CALL_DELAY_S = 8               # seconds between generation calls (rate-limit buffer)

_client = None

def _get_client():
    global _client
    if _client is None:
        if not _API_KEY:
            raise EnvironmentError("Set EVAL_GEMINI_API_KEY or GEMINI_API_KEY in .env")
        _client = genai.Client(api_key=_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """You are an expert evaluator of Arabic legal question-answering systems.

Evaluate the following answer to a legal question. Score it on THREE dimensions,
each from 1 (worst) to 5 (best):

1. **Faithfulness** — Does the answer ONLY use information from the provided context?
   (5 = fully grounded, no hallucination; 1 = invents facts not in context)

2. **Relevance** — Does the answer directly address the question asked?
   (5 = perfectly on-topic; 1 = completely off-topic)

3. **Completeness** — Is the answer thorough and covers all key aspects of the question?
   (5 = comprehensive; 1 = very partial or superficial)

Respond ONLY with a JSON object (no extra text):
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "reasoning": "<one sentence>"}}

---

QUESTION:
{question}

CONTEXT (articles provided to the model):
{context}

ANSWER TO EVALUATE:
{answer}
"""


def _call_judge(question: str, context: str, answer: str,
                max_retries: int = 4) -> dict:
    """Call Gemini judge and parse the JSON score."""
    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context[:2000],   # cap to avoid token overflow
        answer=answer[:1500],
    )
    client = _get_client()
    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=256,
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=config,
            )
            raw = response.text.strip()
            # Extract JSON block even if wrapped in markdown code fences
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"faithfulness": 0, "relevance": 0, "completeness": 0,
                    "reasoning": "parse error"}
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err)
                wait = int(m.group(1)) + 3 if m else 30 * (2 ** attempt)
                print(f"  [Judge] Rate-limited, waiting {wait}s…")
                time.sleep(wait)
            else:
                return {"faithfulness": 0, "relevance": 0, "completeness": 0,
                        "reasoning": f"error: {e}"}
    return {"faithfulness": 0, "relevance": 0, "completeness": 0,
            "reasoning": "max retries"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_generation(
    model_name: str,
    rag_fn,             # callable(question: str) -> {"answer": str, "context": str}
    test_cases: list[TestCase],
    verbose: bool = True,
) -> dict:
    """
    Evaluate generation quality for one model over all test cases.

    Args:
        model_name: Display name for the model.
        rag_fn:     Callable that takes a question and returns
                    {"answer": str, "context": str}.
                    "context" is the text of the retrieved articles (for the judge).
        test_cases: List of TestCase objects.
        verbose:    Whether to print per-question scores.

    Returns:
        {
          "model":        str,
          "faithfulness": float,   ← averaged 1-5
          "relevance":    float,
          "completeness": float,
          "gen_score":    float,   ← average of the three, normalised to 0-1
          "per_case":     list[dict]
        }
    """
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Generation eval — {model_name}  ({len(test_cases)} queries)")
        print(f"{'─'*60}")

    per_case = []

    for tc in test_cases:
        try:
            result = rag_fn(tc.question)
            answer  = result.get("answer",  "")
            context = result.get("context", "")
        except Exception as e:
            if verbose:
                print(f"  X [{tc.id}] RAG error: {e}")
            answer  = ""
            context = ""

        # Brief pause between generation calls to stay within per-minute limits
        time.sleep(GEN_CALL_DELAY_S)

        if not answer:
            scores = {"faithfulness": 1, "relevance": 1, "completeness": 1,
                      "reasoning": "no answer produced"}
        else:
            scores = _call_judge(tc.question, context, answer)

        f = scores.get("faithfulness", 0)
        r = scores.get("relevance",    0)
        c = scores.get("completeness", 0)

        entry = {
            "tc_id":          tc.id,
            "question":       tc.question,
            "domain":         tc.domain,
            "answer_preview": answer[:200],
            "faithfulness":   f,
            "relevance":      r,
            "completeness":   c,
            "reasoning":      scores.get("reasoning", ""),
        }
        per_case.append(entry)

        if verbose:
            avg = (f + r + c) / 3
            print(f"  [{tc.id}] F={f} R={r} C={c} avg={avg:.2f}"
                  f"  | {tc.question[:40]}…")

    n = len(per_case) or 1
    avg_f = sum(e["faithfulness"]  for e in per_case) / n
    avg_r = sum(e["relevance"]     for e in per_case) / n
    avg_c = sum(e["completeness"]  for e in per_case) / n
    gen_score = (avg_f + avg_r + avg_c) / 15  # normalise to 0-1

    agg = {
        "model":        model_name,
        "faithfulness": round(avg_f, 3),
        "relevance":    round(avg_r, 3),
        "completeness": round(avg_c, 3),
        "gen_score":    round(gen_score, 4),
        "per_case":     per_case,
    }

    if verbose:
        print(f"\n  → Faithfulness={avg_f:.2f}/5  "
              f"Relevance={avg_r:.2f}/5  "
              f"Completeness={avg_c:.2f}/5  "
              f"GenScore={gen_score:.1%}")

    return agg
