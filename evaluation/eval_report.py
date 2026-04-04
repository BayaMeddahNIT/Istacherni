"""
eval_report.py
--------------
Reads raw_results.json and prints a clean comparison table,
then saves a report.md and report.json with final accuracy scores.

Accuracy formula:
  accuracy = Recall@5 × 0.40  +  GenScore × 0.60
  (retrieval worth 40%, generation quality worth 60%)
  → expressed as 0–100%
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Column widths
_W = 20


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _bar(v: float, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    filled = int(round(v * width))
    return "█" * filled + "░" * (width - filled)


def compute_accuracy(ret: dict, gen: dict | None) -> float:
    """
    Weighted accuracy score combining retrieval and generation quality.
      60% weight → Recall@5 (did we find the right article?)
      40% weight → Generation score (was the answer good?) — 0 if retrieval-only
    """
    recall5   = ret.get("recall_at_5", 0) if ret else 0
    gen_score = gen.get("gen_score",   0) if gen else 0

    if gen:
        return recall5 * 0.40 + gen_score * 0.60
    else:
        return recall5  # retrieval-only mode


def print_report(all_results: dict, retrieval_only: bool = False):
    """Print a rich console comparison table."""
    has_gen = not retrieval_only and any(
        "generation" in v for v in all_results.values()
    )

    models = list(all_results.keys())
    display = {k: all_results[k].get("model", k) for k in models}

    # Shorten display names for table
    short = {
        k: v.replace("RAG", "").replace("(Dense)", "Dense")
              .replace("(Lexical)", "BM25").strip()
        for k, v in display.items()
    }

    sep = "╠" + "╬".join(["═" * (_W + 2)] * (len(models) + 1)) + "╣"
    top = "╔" + "╦".join(["═" * (_W + 2)] * (len(models) + 1)) + "╗"
    bot = "╚" + "╩".join(["═" * (_W + 2)] * (len(models) + 1)) + "╝"

    def row(label, values: list[str]) -> str:
        label_col = f" {label:<{_W}} "
        val_cols  = "".join(f" {str(v):^{_W}} " for v in values)
        return "║" + label_col + "║" + "║".join(
            f" {str(v):^{_W}} " for v in values
        ) + "║"

    print("\n" + "="*70)
    print("  RAG ARCHITECTURE COMPARISON REPORT")
    print("="*70)

    print(top)
    print(row("Metric", [short[k] for k in models]))
    print(sep)
    print(row("Recall @ 1",
              [_pct(all_results[k]["retrieval"]["recall_at_1"]) for k in models]))
    print(row("Recall @ 3",
              [_pct(all_results[k]["retrieval"]["recall_at_3"]) for k in models]))
    print(row("Recall @ 5",
              [_pct(all_results[k]["retrieval"]["recall_at_5"]) for k in models]))
    print(row("MRR",
              [f"{all_results[k]['retrieval']['mrr']:.3f}" for k in models]))

    if has_gen:
        print(sep)
        print(row("Faithfulness /5",
                  [f"{all_results[k]['generation']['faithfulness']:.2f}" for k in models]))
        print(row("Relevance /5",
                  [f"{all_results[k]['generation']['relevance']:.2f}" for k in models]))
        print(row("Completeness /5",
                  [f"{all_results[k]['generation']['completeness']:.2f}" for k in models]))

    print(sep)
    accuracies = {
        k: compute_accuracy(
            all_results[k].get("retrieval"),
            all_results[k].get("generation"),
        )
        for k in models
    }
    print(row("★ ACCURACY",
              [_pct(accuracies[k]) for k in models]))
    print(bot)

    # Winner
    winner_key = max(accuracies, key=accuracies.__getitem__)
    winner_name = display[winner_key]
    print(f"\n  🏆 WINNER: {winner_name}  ({_pct(accuracies[winner_key])})")

    # Bar chart
    print("\n  Accuracy Bar Chart:")
    for k in models:
        bar = _bar(accuracies[k])
        print(f"  {short[k]:<18s} {bar} {_pct(accuracies[k])}")

    print()
    return accuracies


def save_report(all_results: dict, accuracies: dict, retrieval_only: bool = False):
    """Save report in Markdown and JSON formats."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(all_results.keys())
    has_gen = not retrieval_only and any(
        "generation" in v for v in all_results.values()
    )

    # ── Markdown ───────────────────────────────────────────────────────────────
    lines = [
        "# RAG Architecture Evaluation Report\n",
        "## Summary\n",
        "| Model | Recall@1 | Recall@3 | Recall@5 | MRR |"
        + (" Faithfulness | Relevance | Completeness |" if has_gen else "")
        + " **Accuracy** |",
        "|---|---|---|---|---|"
        + ("---|---|---|" if has_gen else "")
        + "---|",
    ]
    for k in models:
        ret = all_results[k]["retrieval"]
        gen = all_results[k].get("generation", {})
        name = all_results[k].get("model", k)
        acc  = accuracies[k]
        line = (
            f"| {name} "
            f"| {_pct(ret['recall_at_1'])} "
            f"| {_pct(ret['recall_at_3'])} "
            f"| {_pct(ret['recall_at_5'])} "
            f"| {ret['mrr']:.3f} |"
        )
        if has_gen and gen:
            line += (
                f" {gen.get('faithfulness',0):.2f} "
                f"| {gen.get('relevance',0):.2f} "
                f"| {gen.get('completeness',0):.2f} |"
            )
        line += f" **{_pct(acc)}** |"
        lines.append(line)

    winner_key  = max(accuracies, key=accuracies.__getitem__)
    winner_name = all_results[winner_key].get("model", winner_key)
    lines += [
        f"\n## 🏆 Winner\n",
        f"**{winner_name}** with an accuracy of **{_pct(accuracies[winner_key])}**\n",
        "\n## Accuracy Formula\n",
        "```\naccuracy = Recall@5 × 0.40  +  GenScore × 0.60\n"
        if has_gen else
        "```\naccuracy = Recall@5  (retrieval-only mode)\n",
        "```\n",
    ]
    md_path = RESULTS_DIR / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Report saved → {md_path}")

    # ── JSON summary ───────────────────────────────────────────────────────────
    summary = {
        k: {
            "model":       all_results[k].get("model", k),
            "recall_at_1": all_results[k]["retrieval"]["recall_at_1"],
            "recall_at_3": all_results[k]["retrieval"]["recall_at_3"],
            "recall_at_5": all_results[k]["retrieval"]["recall_at_5"],
            "mrr":         all_results[k]["retrieval"]["mrr"],
            "faithfulness":  all_results[k].get("generation", {}).get("faithfulness"),
            "relevance":     all_results[k].get("generation", {}).get("relevance"),
            "completeness":  all_results[k].get("generation", {}).get("completeness"),
            "accuracy":    round(accuracies[k], 4),
        }
        for k in models
    }
    json_path = RESULTS_DIR / "report.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"  📊 JSON saved   → {json_path}")


def generate_report(raw_results_path: Path | None = None,
                    retrieval_only: bool = False):
    """Load raw results from disk and generate the full report."""
    if raw_results_path is None:
        raw_results_path = RESULTS_DIR / "raw_results.json"

    with open(raw_results_path, encoding="utf-8") as f:
        all_results = json.load(f)

    accuracies = print_report(all_results, retrieval_only=retrieval_only)
    save_report(all_results, accuracies, retrieval_only=retrieval_only)
    return all_results, accuracies
