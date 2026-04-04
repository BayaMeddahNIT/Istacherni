"""
run_evaluation.py
-----------------
Single entry point for the RAG Evaluation Pipeline.

Usage:
  # Full evaluation (retrieval + LLM-as-a-judge generation scoring):
  python run_evaluation.py

  # Retrieval only (zero API calls, instant results):
  python run_evaluation.py --retrieval-only

  # Use only 10 questions (save API quota):
  python run_evaluation.py --questions 10

  # Run specific models only:
  python run_evaluation.py --models bm25,graph

  # Combine flags:
  python run_evaluation.py --retrieval-only --models bm25,agentic,graph

  # Just reprint report from existing raw_results.json (no re-running):
  python run_evaluation.py --report-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline — Compare Standard/BM25/Agentic/Graph RAG"
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip generation scoring (no Gemini API calls for generation or judge)."
    )
    parser.add_argument(
        "--questions", type=int, default=20,
        help="Number of test questions to use (1-20, default: 20)."
    )
    parser.add_argument(
        "--models", type=str, default="standard,bm25,agentic,graph",
        help="Comma-separated list of models to evaluate. "
             "Options: standard, bm25, agentic, graph."
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip evaluation; just reprint the report from the last run."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-question verbose output."
    )
    args = parser.parse_args()

    # ── Report-only mode ──────────────────────────────────────────────────────
    if args.report_only:
        print("\n[Eval] Report-only mode — loading existing results…")
        from evaluation.eval_report import generate_report
        raw_path = Path("evaluation/results/raw_results.json")
        if not raw_path.exists():
            print(f"  ❌ No raw results found at {raw_path}. Run evaluation first.")
            sys.exit(1)
        generate_report(raw_path, retrieval_only=args.retrieval_only)
        return

    # ── Parse model list ─────────────────────────────────────────────────────
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    valid  = {"standard", "bm25", "agentic", "graph"}
    bad    = [m for m in models if m not in valid]
    if bad:
        print(f"  ❌ Unknown model(s): {bad}. Valid: {sorted(valid)}")
        sys.exit(1)

    n_questions = max(1, min(args.questions, 20))
    verbose     = not args.quiet

    print("\n" + "="*70)
    print("  RAG EVALUATION PIPELINE")
    print("="*70)
    print(f"  Models        : {', '.join(models)}")
    print(f"  Questions     : {n_questions}")
    print(f"  Mode          : {'Retrieval-only (no API calls)' if args.retrieval_only else 'Full (retrieval + generation + judge)'}")
    print("="*70)

    # ── Load test set ──────────────────────────────────────────────────────────
    from evaluation.eval_testset import load_test_cases
    test_cases = load_test_cases(n=n_questions)

    # ── Run evaluation ────────────────────────────────────────────────────────
    from evaluation.eval_runner import run_all
    all_results = run_all(
        test_cases      = test_cases,
        models          = models,
        retrieval_only  = args.retrieval_only,
        verbose         = verbose,
    )

    # ── Generate report ────────────────────────────────────────────────────────
    from evaluation.eval_report import generate_report
    generate_report(retrieval_only=args.retrieval_only)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
