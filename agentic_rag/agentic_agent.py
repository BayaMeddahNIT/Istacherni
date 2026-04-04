"""
agentic_agent.py
----------------
The core Agentic RAG engine.

How it works — unlike standard RAG:
  ┌──────────────────────────────────────────────────────────────────┐
  │  User question                                                   │
  │       ↓                                                          │
  │  Gemini (THINK) → decides which tool(s) to call                  │
  │       ↓                                                          │
  │  Tool execution  (search_articles / filter_by_domain /           │
  │                   get_article_by_id)                             │
  │       ↓                                                          │
  │  Gemini reviews results → may call MORE tools (iterative)        │
  │       ↓  (loop until Gemini stops calling tools)                 │
  │  Gemini generates the final Arabic answer                        │
  └──────────────────────────────────────────────────────────────────┘

The agent is powered by Gemini 2.5 Flash function-calling.
Uses AGENTIC_GEMINI_API_KEY if set in .env, falls back to GEMINI_API_KEY.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from agentic_rag.agent_knowledge_base import KB

# ── Env ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_API_KEY = os.getenv("AGENTIC_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL    = "gemini-2.5-flash"
MAX_TOOL_ROUNDS = 5   # Maximum agentic iterations before forcing a final answer


# ═══════════════════════════════════════════════════════════════════════
# ── Tool declarations  (sent to Gemini's function-calling API) ────────
# ═══════════════════════════════════════════════════════════════════════

_TOOL_DECLARATIONS = [
    genai_types.FunctionDeclaration(
        name="search_articles",
        description=(
            "Search the full Algerian law corpus using BM25 keyword matching. "
            "Use this as the first tool when you are unsure which law domain "
            "the question belongs to."
        ),
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "query": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="The search query in Arabic (key legal terms from the user's question)."
                ),
                "top_k": genai_types.Schema(
                    type=genai_types.Type.INTEGER,
                    description="Number of articles to return (default 5, max 10)."
                ),
            },
            required=["query"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="filter_by_domain",
        description=(
            "Search within a specific Algerian law domain. "
            "Use when you know the relevant domain (e.g. 'Penal Law', 'Civil Law', "
            "'Labor Law', 'Commercial Law', 'Administrative Law'). "
            "More precise than search_articles when the domain is clear."
        ),
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "domain": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description=(
                        "The law domain to filter by. Examples: "
                        "'Penal Law', 'Civil Law', 'Labor Law', 'Commercial Law'."
                    )
                ),
                "query": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="The search query in Arabic."
                ),
                "top_k": genai_types.Schema(
                    type=genai_types.Type.INTEGER,
                    description="Number of articles to return (default 5, max 10)."
                ),
            },
            required=["domain", "query"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="get_article_by_id",
        description=(
            "Retrieve the full text of a specific article by its ID "
            "(e.g. 'DZ_PENAL_ART_350'). Use this to fetch the complete "
            "details of an article you already know about."
        ),
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "article_id": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="The exact article ID (e.g. 'DZ_PENAL_ART_350')."
                ),
            },
            required=["article_id"],
        ),
    ),
]

_TOOLS = [genai_types.Tool(function_declarations=_TOOL_DECLARATIONS)]


# ═══════════════════════════════════════════════════════════════════════
# ── Tool executor ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _format_article(art: dict) -> dict:
    """Return a clean JSON-serialisable summary of an article for the agent."""
    return {
        "id":                         art.get("id"),
        "law_name":                   art.get("law_name"),
        "law_domain":                 art.get("law_domain"),
        "article_number":             art.get("article_number"),
        "title":                      art.get("title"),
        "text":                       art.get("text_original", "")[:600],  # keep tokens low
        "legal_conditions_summary":   art.get("legal_conditions_summary"),
        "penalties_summary":          art.get("penalties_summary"),
        "keywords":                   art.get("keywords", []),
        "bm25_score":                 art.get("bm25_score"),
    }


def _execute_tool(name: str, args: dict) -> str:
    """
    Dispatch a function-call from Gemini to the actual KB tool.
    Returns a JSON string to send back as the tool result.
    """
    try:
        if name == "search_articles":
            results = KB.search_articles(
                query=args.get("query", ""),
                top_k=min(int(args.get("top_k", 5)), 10),
            )
            payload = [_format_article(a) for a in results]
            return json.dumps({"articles": payload, "count": len(payload)}, ensure_ascii=False)

        elif name == "filter_by_domain":
            results = KB.filter_by_domain(
                domain=args.get("domain", ""),
                query=args.get("query", ""),
                top_k=min(int(args.get("top_k", 5)), 10),
            )
            payload = [_format_article(a) for a in results]
            return json.dumps({"articles": payload, "count": len(payload)}, ensure_ascii=False)

        elif name == "get_article_by_id":
            art = KB.get_article_by_id(args.get("article_id", ""))
            if art:
                return json.dumps({"article": _format_article(art)}, ensure_ascii=False)
            return json.dumps({"error": "Article not found."}, ensure_ascii=False)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════
# ── Agent  ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

_SYSTEM_INSTRUCTION = """أنت محامٍ رقمي متخصص في القانون الجزائري.
لديك أدوات للبحث في قاعدة بيانات المواد القانونية الجزائرية.

إرشادات العمل:
1. حلّل سؤال المستخدم وحدّد أي مجال قانوني ينتمي إليه (جزائي / مدني / عمل / تجاري…).
2. استخدم الأدوات المتاحة للبحث عن المواد ذات الصلة. يمكنك استدعاء أدوات متعددة إذا لزم الأمر.
3. بعد الحصول على المواد الكافية، صِغ إجابة قانونية شاملة باللغة العربية.
4. استشهد برقم المادة واسم القانون لكل معلومة.
5. لا تخترع معلومات خارج المواد المسترجعة.
6. إذا لم تجد مواد كافية بعد البحث، قل ذلك صراحةً."""


def _get_client() -> genai.Client:
    if not _API_KEY:
        raise EnvironmentError(
            "No Gemini API key found. Set AGENTIC_GEMINI_API_KEY or GEMINI_API_KEY in .env"
        )
    return genai.Client(api_key=_API_KEY)


def agentic_answer(question: str, verbose: bool = True) -> dict:
    """
    Run the full agentic RAG loop for a given question.

    Args:
        question: The user's legal question (Arabic or French).
        verbose:  Print the agent's reasoning steps to stdout.

    Returns:
        {
          "answer":       str,           ← the final Arabic answer
          "tools_called": list[dict],    ← log of every tool call + result summary
          "rounds":       int,           ← number of agentic iterations
        }
    """
    client = _get_client()
    tools_log: list[dict] = []

    # ── Conversation history ───────────────────────────────────────────
    history: list[genai_types.Content] = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"{_SYSTEM_INSTRUCTION}\n\n---\n\nسؤال المستخدم:\n{question}")]
        )
    ]

    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2048,
        tools=_TOOLS,
    )

    # ── Agentic loop ──────────────────────────────────────────────────
    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        if verbose:
            print(f"\n  [Agent Round {round_num}] Calling Gemini…")

        # Call Gemini with retry on 429
        response = _call_with_retry(client, history, config)

        candidate = response.candidates[0]
        tool_calls = [
            part for part in candidate.content.parts
            if part.function_call is not None
        ]

        # No tool calls → agent has the final answer
        if not tool_calls:
            final_text = "".join(
                part.text for part in candidate.content.parts
                if hasattr(part, "text") and part.text
            ).strip()
            if verbose:
                print(f"  [Agent Round {round_num}] ✓ Final answer generated.")
            return {
                "answer":       final_text,
                "tools_called": tools_log,
                "rounds":       round_num,
            }

        # ── Execute all tool calls in this round ──────────────────────
        # Add the model's response (with tool calls) to history
        history.append(candidate.content)

        tool_response_parts: list[genai_types.Part] = []
        for part in tool_calls:
            fc   = part.function_call
            name = fc.name
            args = dict(fc.args) if fc.args else {}

            if verbose:
                print(f"  [Agent]  → Tool: {name}({', '.join(f'{k}={repr(v)}' for k,v in args.items())})")

            result_str = _execute_tool(name, args)
            result_obj = json.loads(result_str)

            # Log it
            tools_log.append({
                "round":  round_num,
                "tool":   name,
                "args":   args,
                "result_summary": (
                    f"{result_obj.get('count', 1)} article(s) found"
                    if "count" in result_obj else
                    result_obj.get("error", "ok")
                ),
            })

            tool_response_parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=name,
                        response={"result": result_str},
                    )
                )
            )

        # Add all tool results as a single "tool" role message
        history.append(
            genai_types.Content(role="tool", parts=tool_response_parts)
        )

    # ── Reached MAX_TOOL_ROUNDS — force a final generation ────────────
    if verbose:
        print(f"  [Agent] Max rounds reached. Forcing final answer…")

    history.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(
                text="بناءً على ما استرجعته من المواد القانونية، قدّم الآن إجابتك النهائية الشاملة."
            )]
        )
    )
    final_config = genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2048,
    )
    response = _call_with_retry(client, history, final_config)
    final_text = "".join(
        part.text for part in response.candidates[0].content.parts
        if hasattr(part, "text") and part.text
    ).strip()

    return {
        "answer":       final_text,
        "tools_called": tools_log,
        "rounds":       MAX_TOOL_ROUNDS,
    }


def _call_with_retry(client, history, config, max_retries: int = 4):
    """Call Gemini with exponential-backoff retry on 429."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err)
                wait = int(m.group(1)) + 3 if m else 30 * (2 ** attempt)
                print(f"  [Agent] Rate-limited, waiting {wait}s…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("_call_with_retry: all retries exhausted")
