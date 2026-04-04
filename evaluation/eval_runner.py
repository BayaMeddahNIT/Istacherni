"""
eval_runner.py
--------------
Wires up all 4 RAG models and runs retrieval + generation evaluation.

Each model is wrapped in a thin adapter that provides:
  • retriever_fn(query, top_k) -> list[dict with 'id']
  • rag_fn(question)           -> {"answer": str, "context": str}

Standard RAG is loaded read-only from backend/ (no files modified).
If ChromaDB is unavailable (not indexed yet), Standard RAG is skipped gracefully.
"""

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════════
# ── Model adapters ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Adapter 1: Standard RAG (ChromaDB + Dense Embeddings) ─────────────────────

def _make_standard_rag():
    """Returns (retriever_fn, rag_fn) for Standard RAG, or None if unavailable."""
    try:
        from backend.rag.retrieval.retriever  import retrieve
        from backend.rag.generation.generator import generate_answer

        def retriever_fn(query: str, top_k: int) -> list[dict]:
            results = retrieve(query, top_k=top_k)
            # Normalise: standard RAG returns {id, text, metadata, score}
            return [
                {
                    "id":    r["id"],
                    "text_original": r.get("metadata", {}).get("text_original", r.get("text", "")),
                    "law_name":      r.get("metadata", {}).get("law_name", ""),
                    "article_number": r.get("metadata", {}).get("article_number", ""),
                    "score": r.get("score", 0),
                }
                for r in results
            ]

        def rag_fn(question: str) -> dict:
            chunks = retrieve(question, top_k=5)
            answer = generate_answer(question, chunks)
            context = "\n---\n".join(
                c.get("metadata", {}).get("text_original", c.get("text", ""))
                for c in chunks
            )
            return {"answer": answer, "context": context}

        print("  [Runner] Standard RAG: ✓ loaded")
        return retriever_fn, rag_fn

    except Exception as e:
        print(f"  [Runner] Standard RAG: ✗ skipped ({e})")
        return None, None


# ── Adapter 2: BM25 RAG ─────────────────────────────────────────────────────

def _make_bm25_rag():
    from bm25_rag.bm25_retriever import bm25_retrieve
    from bm25_rag.bm25_generator  import bm25_generate

    def retriever_fn(query: str, top_k: int) -> list[dict]:
        return bm25_retrieve(query, top_k=top_k)

    def rag_fn(question: str) -> dict:
        chunks = bm25_retrieve(question, top_k=5)
        answer = bm25_generate(question, chunks)
        context = "\n---\n".join(c.get("text_original", "") for c in chunks)
        return {"answer": answer, "context": context}

    print("  [Runner] BM25 RAG: ✓ loaded")
    return retriever_fn, rag_fn


# ── Adapter 3: Agentic RAG ──────────────────────────────────────────────────

def _make_agentic_rag():
    # For RETRIEVAL evaluation: Agentic RAG uses BM25 as its underlying search
    # mechanism (via the KB tools). We reuse the already-built bm25_rag index
    # so results are directly comparable and reliably correct.
    from bm25_rag.bm25_retriever import bm25_retrieve
    from agentic_rag.agentic_agent import agentic_answer

    def retriever_fn(query: str, top_k: int) -> list[dict]:
        # Returns same-format dicts as BM25 (with 'id' field matching corpus)
        try:
            results = bm25_retrieve(query, top_k=top_k)
            if not results:
                print(f"  [Agentic/Retrieval] WARNING: no results for '{query[:40]}'")
            return results
        except Exception as e:
            print(f"  [Agentic/Retrieval] ERROR: {type(e).__name__}: {e}")
            return []

    def rag_fn(question: str) -> dict:
        try:
            result  = agentic_answer(question, verbose=False)
            answer  = result.get("answer", "")
            n_tools = len(result.get("tools_called", []))
            n_rounds = result.get("rounds", 0)
            context = f"[Agentic: {n_rounds} round(s), {n_tools} tool call(s)]"
            return {"answer": answer, "context": context}
        except Exception as e:
            print(f"  [Agentic/Gen] ERROR: {type(e).__name__}: {e}")
            return {"answer": "", "context": ""}

    print("  [Runner] Agentic RAG: loaded (retrieval via BM25 index)")
    return retriever_fn, rag_fn


# ── Adapter 4: Graph RAG ────────────────────────────────────────────────────

def _make_graph_rag():
    from graph_rag.graph_retriever import graph_retrieve
    from graph_rag.graph_generator  import graph_generate

    def retriever_fn(query: str, top_k: int) -> list[dict]:
        return graph_retrieve(query, top_k=top_k)

    def rag_fn(question: str) -> dict:
        chunks  = graph_retrieve(question, top_k=5)
        answer  = graph_generate(question, chunks)
        context = "\n---\n".join(c.get("text_original", "") for c in chunks)
        return {"answer": answer, "context": context}

    print("  [Runner] Graph RAG: loaded")
    return retriever_fn, rag_fn


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main runner ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(
    test_cases,
    models: list[str] = None,
    retrieval_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full evaluation for all (or selected) models.

    Args:
        test_cases:     List of TestCase from eval_testset.load_test_cases().
        models:         Which models to run. None = all four.
                        Options: ["standard", "bm25", "agentic", "graph"]
        retrieval_only: If True, skip generation scoring (no API calls).
        verbose:        Print progress.

    Returns:
        Raw results dict saved to evaluation/results/raw_results.json
    """
    from evaluation.eval_retrieval  import evaluate_retrieval
    from evaluation.eval_generation import evaluate_generation

    if models is None:
        models = ["standard", "bm25", "agentic", "graph"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model adapters ─────────────────────────────────────────────────
    print("\n[Runner] Loading models…")
    loaders = {
        "standard": _make_standard_rag,
        "bm25":     _make_bm25_rag,
        "agentic":  _make_agentic_rag,
        "graph":    _make_graph_rag,
    }
    display_names = {
        "standard": "Standard RAG (Dense)",
        "bm25":     "BM25 RAG (Lexical)",
        "agentic":  "Agentic RAG",
        "graph":    "Graph RAG",
    }

    all_results = {}

    for key in models:
        name = display_names.get(key, key)
        try:
            ret_fn, rag_fn = loaders[key]()
        except Exception as e:
            print(f"  [Runner] {name}: ✗ load failed — {e}")
            continue

        if ret_fn is None:
            continue

        model_result = {"model": name}

        # ── Retrieval ────────────────────────────────────────────────────────
        t0 = time.time()
        ret_result = evaluate_retrieval(name, ret_fn, test_cases, verbose=verbose)
        model_result["retrieval"] = ret_result
        model_result["retrieval_time_s"] = round(time.time() - t0, 1)

        # ── Generation (skip if retrieval_only) ──────────────────────────────
        if not retrieval_only and rag_fn is not None:
            t0 = time.time()
            gen_result = evaluate_generation(name, rag_fn, test_cases, verbose=verbose)
            model_result["generation"] = gen_result
            model_result["generation_time_s"] = round(time.time() - t0, 1)

        all_results[key] = model_result

    # ── Save raw results to disk ─────────────────────────────────────────────
    out_path = RESULTS_DIR / "raw_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Runner] Raw results saved → {out_path}")

    return all_results
