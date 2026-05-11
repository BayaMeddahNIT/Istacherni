# -*- coding: utf-8 -*-
"""
generate_eval_answers.py
------------------------
Runs every question in questions.txt through the three hybrid RAG pipelines
(QWEN, CamelBERT, BGE-M3) and saves the answers to individual .txt files
inside the evaluation/ directory.

Output format (one block per question):
    [User]: <question>
    ANSWER:
    <answer>
    (Time taken: X.XX seconds)
    SOURCES:
    [1] <law_name> - المادة <article_number>
    [2] ...
    ════════════════════════════════════════

Usage:
    # Run all three pipelines:
    python generate_eval_answers.py

    # Run a single pipeline:
    python generate_eval_answers.py --model qwen
    python generate_eval_answers.py --model camelbert
    python generate_eval_answers.py --model bge

    # Resume from question index N (0-based):
    python generate_eval_answers.py --model bge --start 10

    # Use a different questions file:
    python generate_eval_answers.py --questions my_questions.txt

    # Limit to first N questions (useful for quick smoke-tests):
    python generate_eval_answers.py --limit 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ── Make sure the project root is on sys.path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding so Arabic + symbols print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

SEPARATOR = "═" * 60


# ══════════════════════════════════════════════════════════════════════════════
# ── Question parser ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _is_arabic(text: str) -> bool:
    """Return True if the line contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _is_section_header(text: str) -> bool:
    """
    Return True for category header lines, e.g.:
        '1. Penal Law (قانون العقوبات) – 25 questions'
        '🟦 2. Civil Law …'
    """
    stripped = text.strip()
    # Starts with a digit + dot OR an emoji block indicator
    if re.match(r"^\d+\.", stripped):
        return True
    # Emoji block characters (🟦 🟨 🟩 🟪 etc.)
    if stripped and ord(stripped[0]) > 0x1F000:
        return True
    return False


def load_questions(filepath: Path) -> list[str]:
    """
    Parse questions.txt and return only the Arabic questions,
    stripping headers, blank lines, and section titles.
    """
    questions: list[str] = []
    with open(filepath, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if _is_section_header(line):
                continue
            if _is_arabic(line):
                questions.append(line)
    return questions


# ══════════════════════════════════════════════════════════════════════════════
# ── Output formatter ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def format_entry(
    question: str,
    answer: str,
    sources: list[dict],
    elapsed: float,
) -> str:
    """
    Format one QA pair into the canonical evaluation text block.

    Example output:
        [User]: هل النصب في التجارة يُعاقب عليه القانون؟
        ANSWER:
        نعم، …
        (Time taken: 30.33 seconds)
        SOURCES:
        [1] قانون العقوبات - المادة 372
        [2] قانون العقوبات - المادة 383
        ════════════════════════════════════════
    """
    lines: list[str] = []
    lines.append(f"[User]: {question}")
    lines.append("ANSWER:")
    lines.append(answer.strip())
    lines.append(f"(Time taken: {elapsed:.2f} seconds)")
    lines.append("SOURCES:")
    if sources:
        for idx, src in enumerate(sources, start=1):
            law  = src.get("law_name", "")
            art  = src.get("article_number", "")
            lines.append(f"[{idx}] {law} - المادة {art}")
    else:
        lines.append("[غير متوفر]")
    lines.append(SEPARATOR)
    return "\n".join(lines) + "\n\n"


# ══════════════════════════════════════════════════════════════════════════════
# ── Pipeline wrappers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_qwen_hybrid(question: str, top_k: int = 7) -> tuple[str, list[dict]]:
    """
    QWEN Hybrid RAG:
      Retrieval  → Qwen3-Embedding-8B (dense) via ChromaDB
      Generation → qwen2.5:7b via Ollama
    """
    from qwen_rag.qwen_retriever import qwen_retrieve
    from qwen_rag.qwen_generator  import qwen_generate

    chunks = qwen_retrieve(question, top_k=top_k)
    answer = qwen_generate(question, chunks)
    return answer, chunks


def run_camelbert_hybrid(question: str, top_k: int = 7) -> tuple[str, list[dict]]:
    """
    CamelBERT Hybrid RAG:
      Retrieval  → CamelBERT (dense) + BM25 (sparse) via RRF
      Generation → graph_generate (qwen2.5:7b via Ollama)
    """
    from camelbert_rag.hybrid_retriever       import hybrid_retrieve
    from graph_rag_local.graph_generator       import graph_generate

    chunks = hybrid_retrieve(question, top_k=top_k)
    answer = graph_generate(question, chunks)
    return answer, chunks


def run_bge_hybrid(question: str, top_k: int = 7) -> tuple[str, list[dict]]:
    """
    BGE-M3 Hybrid RAG:
      Retrieval  → BAAI/bge-m3 (dense) + BM25 (sparse) via RRF
      Generation → graph_generate (qwen2.5:7b via Ollama)
    """
    from hybrid_rag.hybrid_retriever     import hybrid_retrieve
    from graph_rag_local.graph_generator  import graph_generate

    chunks = hybrid_retrieve(question, top_k=top_k)
    answer = graph_generate(question, chunks)
    return answer, chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── Pipeline registry ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

PIPELINES: dict[str, dict] = {
    "qwen": {
        "label":    "QWEN Hybrid RAG  (Qwen3-Embedding + qwen2.5:7b)",
        "fn":       run_qwen_hybrid,
        "out_file": EVAL_DIR / "answers_qwen_hybrid.txt",
    },
    "camelbert": {
        "label":    "CamelBERT Hybrid RAG  (CamelBERT+BM25 RRF + qwen2.5:7b)",
        "fn":       run_camelbert_hybrid,
        "out_file": EVAL_DIR / "answers_camelbert_hybrid.txt",
    },
    "bge": {
        "label":    "BGE-M3 Hybrid RAG  (BGE-M3+BM25 RRF + qwen2.5:7b)",
        "fn":       run_bge_hybrid,
        "out_file": EVAL_DIR / "answers_bge_m3_hybrid.txt",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Core runner ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _count_existing_entries(path: Path) -> int:
    """Count how many [User]: blocks already exist in the output file."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"^\[User\]:", text, re.MULTILINE))


def run_pipeline(
    key: str,
    questions: list[str],
    start: int = 0,
    top_k: int = 7,
    verbose: bool = True,
) -> None:
    """Run one pipeline over all questions and append results to its output file."""
    cfg      = PIPELINES[key]
    fn       = cfg["fn"]
    out_path: Path = cfg["out_file"]
    label    = cfg["label"]

    # ── Resume support: skip already-answered questions ───────────────────────
    already_done = _count_existing_entries(out_path)
    effective_start = max(start, already_done)

    print(f"\n{'━'*65}")
    print(f"  Pipeline : {label}")
    print(f"  Output   : {out_path}")
    print(f"  Questions: {len(questions)}  |  Starting at index: {effective_start}")
    print(f"{'━'*65}")

    if effective_start >= len(questions):
        print("  ✅ All questions already answered — nothing to do.")
        return

    # Open in append mode so we can resume without re-running answered items
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, question in enumerate(questions):
            if i < effective_start:
                continue

            q_num = i + 1
            total = len(questions)
            print(f"  [{q_num}/{total}] {question[:70]}…" if len(question) > 70 else f"  [{q_num}/{total}] {question}")

            t0 = time.perf_counter()
            try:
                answer, chunks = fn(question, top_k=top_k)
                elapsed = time.perf_counter() - t0
                entry = format_entry(question, answer, chunks, elapsed)
                print(f"        OK {elapsed:.2f}s  | {len(chunks)} sources")
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                error_answer = f"[ERROR] {type(exc).__name__}: {exc}"
                entry = format_entry(question, error_answer, [], elapsed)
                print(f"        ERR {elapsed:.2f}s  | ERROR: {exc}")

            fh.write(entry)
            fh.flush()   # write to disk immediately so progress is never lost

    print(f"\n  [DONE] Results saved -> {out_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation answers for Istacherni Hybrid RAG pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        choices=["all", "qwen", "camelbert", "bge"],
        default="all",
        help="Which pipeline to run  (default: all)",
    )
    parser.add_argument(
        "--questions", "-q",
        default=str(PROJECT_ROOT / "questions.txt"),
        help="Path to the questions file  (default: questions.txt)",
    )
    parser.add_argument(
        "--start", "-s",
        type=int, default=0,
        help="0-based question index to start from  (default: 0)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int, default=None,
        help="Max number of questions to process  (default: all)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int, default=7,
        help="Number of articles to retrieve per question  (default: 7)",
    )
    args = parser.parse_args()

    # ── Load questions ────────────────────────────────────────────────────────
    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"❌  Questions file not found: {q_path}")
        sys.exit(1)

    questions = load_questions(q_path)
    if args.limit:
        questions = questions[: args.limit]

    print(f"\n[INFO] Loaded {len(questions)} questions from {q_path.name}")

    # ── Select pipelines to run ───────────────────────────────────────────────
    keys = list(PIPELINES.keys()) if args.model == "all" else [args.model]

    for key in keys:
        run_pipeline(
            key=key,
            questions=questions,
            start=args.start,
            top_k=args.top_k,
        )

    print(f"\n[DONE] All done! Output files are in: {EVAL_DIR}\n")


if __name__ == "__main__":
    main()
