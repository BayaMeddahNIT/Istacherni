"""
qwen_pipeline.py
----------------
End-to-end Qwen RAG pipeline: question → retrieve → generate → answer.

Usage (both forms work):
    python qwen_rag/qwen_pipeline.py
    python -m qwen_rag.qwen_pipeline
"""

from __future__ import annotations

# ── Make "qwen_rag" importable when run as a plain script ─────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict, List, Optional


def qwen_ask(
    question: str,
    top_k: int = 5,
    min_score: float = 0.0,
    return_sources: bool = False,
) -> str | tuple[str, List[Dict[str, Any]]]:
    """
    Full RAG pipeline: embed query → retrieve articles → generate answer.

    Args:
        question:       User's legal question in Arabic (or French).
        top_k:          Number of articles to retrieve.
        min_score:      Minimum cosine similarity for retrieved articles.
        return_sources: If True, return (answer, retrieved_articles) tuple.

    Returns:
        Generated answer string, or (answer, sources) if return_sources=True.
    """
    from qwen_rag.qwen_retriever import qwen_retrieve
    from qwen_rag.qwen_generator import qwen_generate

    retrieved = qwen_retrieve(question, top_k=top_k, min_score=min_score)
    answer    = qwen_generate(question, retrieved)

    if return_sources:
        return answer, retrieved
    return answer


# ── Interactive CLI ────────────────────────────────────────────────────────────

def _interactive():
    """Simple interactive REPL for testing the pipeline."""
    from qwen_rag.qwen_generator import check_ollama_health

    print("\n" + "=" * 65)
    print("  Qwen Legal RAG — Algerian Law")
    print("  Embeddings : Qwen3-Embedding-0.6B  (local)")
    print("  Generator  : Qwen2.5-7B via Ollama")
    print("=" * 65)

    if not check_ollama_health():
        print("\n[!] Fix Ollama setup before continuing.")
        return

    print("\nType your question in Arabic (or 'quit' to exit).\n")

    while True:
        try:
            question = input("سؤال: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question or question.lower() in {"quit", "exit", "خروج"}:
            print("Goodbye.")
            break

        print("\n[Retrieving …]", flush=True)
        answer, sources = qwen_ask(question, top_k=5, return_sources=True)

        print(f"\n{'─' * 65}")
        print("المصادر المسترجعة:")
        for i, s in enumerate(sources, 1):
            print(
                f"  [{i}] {s['law_name']} — المادة {s['article_number']} "
                f"(score={s['score']})"
            )

        print(f"\n{'─' * 65}")
        print("الإجابة:\n")
        print(answer)
        print(f"{'─' * 65}\n")


# ── Batch evaluation helper ────────────────────────────────────────────────────

def batch_evaluate(
    questions: List[str],
    top_k: int = 5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the pipeline over a list of questions and collect results.

    Returns:
        List of dicts: {question, answer, sources, latency_s}
    """
    import time

    results = []
    for i, q in enumerate(questions, 1):
        if verbose:
            print(f"[{i}/{len(questions)}] {q[:60]} …")
        t0 = time.perf_counter()
        answer, sources = qwen_ask(q, top_k=top_k, return_sources=True)
        elapsed = round(time.perf_counter() - t0, 2)
        results.append({
            "question":  q,
            "answer":    answer,
            "sources":   sources,
            "latency_s": elapsed,
        })
        if verbose:
            if sources:
                print(
                    f"     ✓ {elapsed}s  | top: "
                    f"{sources[0]['law_name']} م{sources[0]['article_number']}"
                )
            else:
                print(f"     ✓ {elapsed}s  | (no sources)")
    return results


if __name__ == "__main__":
    _interactive()
