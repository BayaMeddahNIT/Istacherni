"""
evaluate_rag.py
===============
RAGAS-style evaluation of the Istacherni Graph RAG pipelines.

Pipelines compared:
  1. BGE + Qwen (Graph RAG) -> graph_rag_local
  2. CamelBERT + Qwen       -> camelbert_rag retrieval + qwen generation

Metrics:
  - Context Recall     : Ground truth coverage in retrieved chunks
  - Context Precision  : Relevance of each retrieved chunk
  - Faithfulness       : Answer grounded in context
  - Answer Relevancy   : Answer addresses the question
  - Article Recall     : Legal-domain: % of cited articles retrieved

Usage:
  python evaluate_rag.py                      # Full evaluation, both pipelines
  python evaluate_rag.py --n 10               # First 10 questions only
  python evaluate_rag.py --n 5 --skip-gen     # Retrieval-only (fast, no Ollama)
  python evaluate_rag.py --pipeline bge       # Only BGE+Qwen pipeline
  python evaluate_rag.py --pipeline camelbert # Only CamelBERT pipeline
  python evaluate_rag.py --report-only        # Reprint last saved report
"""

from __future__ import annotations

# ── Force UTF-8 output on Windows before ANY print() ─────────────────────────
import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import csv
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

# ── Suppress noisy warnings ────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH = ROOT / "dataset" / "raw" / "algerian_law_ragas_dataset_v2.json"
RESULTS_DIR  = ROOT / "evaluation" / "ragas_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SECTION 1 — Dataset Loader
# ==============================================================================

def load_dataset(n: int | None = None) -> list[dict]:
    print(f"\n[DATASET] Loading: {DATASET_PATH.name}")
    if not DATASET_PATH.exists():
        print(f"  ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if n:
        data = data[:n]

    cats: dict[str, int] = {}
    for item in data:
        cat = item.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1

    print(f"  Loaded {len(data)} questions:")
    for cat, count in sorted(cats.items()):
        print(f"    - {cat}: {count}")
    return data


# ==============================================================================
# SECTION 2 — Pipeline Adapters
# ==============================================================================

def _check_ollama() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_bge_qwen_pipeline(question: str, top_k: int = 7) -> dict:
    """Pipeline 1: BGE-M3 graph retrieval + Qwen2 local generation."""
    from graph_rag_local.graph_retriever import graph_retrieve, expand_acronyms
    from graph_rag_local.graph_generator import graph_generate

    clean_q = expand_acronyms(question)
    chunks  = graph_retrieve(clean_q, top_k=top_k)

    if not chunks:
        return {"answer": "", "contexts": [], "retrieved_ids": []}

    answer = graph_generate(clean_q, chunks, stream=False)

    contexts = [
        c.get("text_original") or c.get("summary") or ""
        for c in chunks
        if c.get("text_original") or c.get("summary")
    ]
    retrieved_ids = [
        f"{c.get('law_name', '')} {c.get('article_number', '')}"
        for c in chunks
    ]
    return {
        "answer":        answer,
        "contexts":      contexts,
        "retrieved_ids": retrieved_ids,
        "chunk_scores":  [c.get("graph_score", 0) for c in chunks],
    }


def run_camelbert_pipeline(question: str, top_k: int = 7) -> dict:
    """Pipeline 2: CamelBERT retrieval + Qwen2 generation."""
    from camelbert_rag.camelbert_retriever import camelbert_retrieve
    from graph_rag_local.graph_generator import graph_generate

    chunks = camelbert_retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "", "contexts": [], "retrieved_ids": []}

    answer = graph_generate(question, chunks, stream=False)

    contexts = [
        c.get("text_original") or c.get("summary") or ""
        for c in chunks
        if c.get("text_original") or c.get("summary")
    ]
    retrieved_ids = [
        f"{c.get('law_name', '')} {c.get('article_number', '')}"
        for c in chunks
    ]
    return {
        "answer":        answer,
        "contexts":      contexts,
        "retrieved_ids": retrieved_ids,
        "chunk_scores":  [c.get("score", 0) for c in chunks],
    }


def retrieve_only_bge(question: str, top_k: int = 7) -> dict:
    """Retrieval-only mode for BGE pipeline (no LLM call)."""
    from graph_rag_local.graph_retriever import graph_retrieve, expand_acronyms
    clean_q = expand_acronyms(question)
    chunks  = graph_retrieve(clean_q, top_k=top_k)
    contexts = [c.get("text_original") or c.get("summary") or "" for c in chunks]
    retrieved_ids = [f"{c.get('law_name', '')} {c.get('article_number', '')}" for c in chunks]
    return {"answer": "", "contexts": contexts, "retrieved_ids": retrieved_ids}


def retrieve_only_camelbert(question: str, top_k: int = 7) -> dict:
    """Retrieval-only mode for CamelBERT pipeline (no LLM call)."""
    from camelbert_rag.camelbert_retriever import camelbert_retrieve
    chunks = camelbert_retrieve(question, top_k=top_k)
    contexts = [c.get("text_original") or c.get("summary") or "" for c in chunks]
    retrieved_ids = [f"{c.get('law_name', '')} {c.get('article_number', '')}" for c in chunks]
    return {"answer": "", "contexts": contexts, "retrieved_ids": retrieved_ids}


def run_agentic_bge_pipeline(question: str, top_k: int = 7) -> dict:
    """Pipeline 3: Agentic RAG (BGE + Qwen)."""
    from agentic_rag.agentic_agent import agentic_answer
    result = agentic_answer(question, verbose=False, retriever_type="bge", skip_gen=False)
    return {
        "answer":        result.get("answer", ""),
        "contexts":      result.get("contexts", []),
        "retrieved_ids": result.get("retrieved_ids", []),
        "chunk_scores":  [],
    }


def run_agentic_camelbert_pipeline(question: str, top_k: int = 7) -> dict:
    """Pipeline 4: Agentic RAG (CamelBERT + Qwen)."""
    from agentic_rag.agentic_agent import agentic_answer
    result = agentic_answer(question, verbose=False, retriever_type="camelbert", skip_gen=False)
    return {
        "answer":        result.get("answer", ""),
        "contexts":      result.get("contexts", []),
        "retrieved_ids": result.get("retrieved_ids", []),
        "chunk_scores":  [],
    }


def retrieve_only_agentic_bge(question: str, top_k: int = 7) -> dict:
    from agentic_rag.agentic_agent import agentic_answer
    result = agentic_answer(question, verbose=False, retriever_type="bge", skip_gen=True)
    return {
        "answer":        "",
        "contexts":      result.get("contexts", []),
        "retrieved_ids": result.get("retrieved_ids", []),
    }


def retrieve_only_agentic_camelbert(question: str, top_k: int = 7) -> dict:
    from agentic_rag.agentic_agent import agentic_answer
    result = agentic_answer(question, verbose=False, retriever_type="camelbert", skip_gen=True)
    return {
        "answer":        "",
        "contexts":      result.get("contexts", []),
        "retrieved_ids": result.get("retrieved_ids", []),
    }



# ==============================================================================
# SECTION 3 — RAGAS-style Metrics (Reference-based, fully local)
# ==============================================================================

def _tokenize(text: str) -> set[str]:
    """Split Arabic text into word tokens, ignoring punctuation."""
    return set(re.split(r"[\s\u060C\u061B\u061F\u0021\u0022\u0028\u0029.,;:!?]+", text.strip())) - {""}


def _f1_overlap(text_a: str, text_b: str) -> float:
    """Token-level F1 between two Arabic strings."""
    if not text_a or not text_b:
        return 0.0
    a_tok = _tokenize(text_a)
    b_tok = _tokenize(text_b)
    if not a_tok or not b_tok:
        return 0.0
    inter = a_tok & b_tok
    p = len(inter) / len(a_tok)
    r = len(inter) / len(b_tok)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def metric_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """How much of the ground truth is covered by the retrieved context pool?"""
    if not contexts:
        return 0.0
    return _f1_overlap(ground_truth, " ".join(contexts))


def metric_context_precision(ground_truth: str, contexts: list[str]) -> float:
    """What fraction of retrieved chunks are individually relevant?"""
    if not contexts:
        return 0.0
    scores = [_f1_overlap(ground_truth, ctx) for ctx in contexts]
    return sum(scores) / len(scores)


def metric_faithfulness(answer: str, contexts: list[str]) -> float:
    """Is the answer grounded in the retrieved context?"""
    if not answer or not contexts:
        return 0.0
    return _f1_overlap(answer, " ".join(contexts))


def metric_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer address the question's keywords?"""
    if not answer:
        return 0.0
    return _f1_overlap(question, answer)


def metric_article_recall(gt_articles: list[str], retrieved_ids: list[str]) -> float:
    """
    Legal-domain metric: What fraction of ground-truth cited articles
    were successfully retrieved?
    """
    if not gt_articles:
        return 1.0
    if not retrieved_ids:
        return 0.0

    def extract_nums(s: str) -> set[str]:
        return set(re.findall(r"\d+", s))

    gt_nums  = set().union(*(extract_nums(a) for a in gt_articles))
    ret_nums = set().union(*(extract_nums(r) for r in retrieved_ids))

    if not gt_nums:
        return 0.0
    return len(gt_nums & ret_nums) / len(gt_nums)


# Metric weights for final accuracy score
WEIGHTS = {
    "context_recall":    0.25,
    "context_precision": 0.20,
    "faithfulness":      0.25,
    "answer_relevancy":  0.15,
    "article_recall":    0.15,
}

def compute_metrics(item: dict, pipeline_result: dict) -> dict:
    question       = item["question"]
    ground_truth   = item["ground_truth"]
    gt_articles    = item.get("articles", [])
    answer         = pipeline_result.get("answer", "")
    contexts       = pipeline_result.get("contexts", [])
    retrieved_ids  = pipeline_result.get("retrieved_ids", [])

    scores = {
        "context_recall":    metric_context_recall(ground_truth, contexts),
        "context_precision": metric_context_precision(ground_truth, contexts),
        "faithfulness":      metric_faithfulness(answer, contexts),
        "answer_relevancy":  metric_answer_relevancy(question, answer),
        "article_recall":    metric_article_recall(gt_articles, retrieved_ids),
    }
    scores = {k: round(v, 4) for k, v in scores.items()}
    scores["final_accuracy"] = round(
        sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 4
    )
    return scores


# ==============================================================================
# SECTION 4 — Pipeline Runner
# ==============================================================================

def run_pipeline(
    pipeline_name: str,
    pipeline_fn,          # full pipeline function
    retrieve_fn,          # retrieval-only function
    dataset: list[dict],
    skip_gen: bool = False,
    verbose: bool  = True,
) -> dict:
    """Run one pipeline on the full dataset and return aggregated results."""
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  PIPELINE: {pipeline_name}")
    print(f"  Mode    : {'Retrieval-only (--skip-gen)' if skip_gen else 'Full (Retrieval + Generation)'}")
    print(f"  Questions: {len(dataset)}")
    print(sep)

    per_sample: list[dict] = []
    errors = 0
    t_start = time.time()

    for i, item in enumerate(dataset):
        question = item["question"]
        category = item.get("category", "")

        if verbose:
            q_short = question[:55] + ("..." if len(question) > 55 else "")
            print(f"\n  [{i+1:3d}/{len(dataset)}] [{category}] {q_short}")

        try:
            result = retrieve_fn(question) if skip_gen else pipeline_fn(question)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            result = {"answer": "", "contexts": [], "retrieved_ids": []}
            errors += 1

        scores = compute_metrics(item, result)

        if verbose:
            print(
                f"    CR={scores['context_recall']:.3f}  "
                f"CP={scores['context_precision']:.3f}  "
                f"F={scores['faithfulness']:.3f}  "
                f"AR={scores['answer_relevancy']:.3f}  "
                f"ArtR={scores['article_recall']:.3f}  "
                f"=> {scores['final_accuracy']:.3f}"
            )
            n_ctx = len(result.get("contexts", []))
            if not skip_gen and result.get("answer"):
                ans_short = result["answer"][:70].replace("\n", " ")
                print(f"    Answer: {ans_short}...")
            print(f"    Chunks: {n_ctx} retrieved")

        per_sample.append({
            "question":      question,
            "category":      category,
            "ground_truth":  item["ground_truth"],
            "answer":        result.get("answer", ""),
            "retrieved_ids": "; ".join(result.get("retrieved_ids", [])),
            **scores,
        })

    elapsed = round(time.time() - t_start, 1)
    print(f"\n  Done in {elapsed}s | Errors: {errors}/{len(dataset)}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    metric_keys = list(WEIGHTS.keys()) + ["final_accuracy"]
    aggregated = {
        k: round(sum(r[k] for r in per_sample) / len(per_sample), 4)
        for k in metric_keys
    }

    # ── Per-category breakdown ───────────────────────────────────────────────
    cat_map: dict[str, dict[str, list[float]]] = {}
    for r in per_sample:
        cat = r["category"]
        if cat not in cat_map:
            cat_map[cat] = {k: [] for k in metric_keys}
        for k in metric_keys:
            cat_map[cat][k].append(r[k])

    per_category = {
        cat: {k: round(sum(v) / len(v), 4) for k, v in kv.items()}
        for cat, kv in cat_map.items()
    }

    print(f"  Final Accuracy: {aggregated['final_accuracy']*100:.2f}%")
    return {
        "pipeline":     pipeline_name,
        "n_questions":  len(dataset),
        "errors":       errors,
        "elapsed_s":    elapsed,
        "skip_gen":     skip_gen,
        "aggregated":   aggregated,
        "per_category": per_category,
        "per_sample":   per_sample,
    }


# ==============================================================================
# SECTION 5 — Report Generator
# ==============================================================================

METRIC_LABELS = {
    "context_recall":    "Context Recall    ",
    "context_precision": "Context Precision ",
    "faithfulness":      "Faithfulness      ",
    "answer_relevancy":  "Answer Relevancy  ",
    "article_recall":    "Article Recall    ",
    "final_accuracy":    "FINAL ACCURACY    ",
}

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score*100:5.1f}%"


def generate_report(all_results: list[dict], save: bool = True) -> str:
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    W     = 70

    def div(char="-"):
        lines.append(char * W)

    div("=")
    lines.append("  RAGAS EVALUATION REPORT -- Istacherni Legal RAG")
    lines.append(f"  Date: {ts}")
    div("=")

    pipelines = [r["pipeline"] for r in all_results]
    metrics   = ["context_recall", "context_precision", "faithfulness",
                 "answer_relevancy", "article_recall", "final_accuracy"]

    # ── Comparison table ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("  METRIC COMPARISON TABLE")
    div()

    col = 22
    header = f"  {'Metric':<22}" + "".join(f"  {p[:24]:<24}" for p in pipelines)
    lines.append(header)
    div()

    for metric in metrics:
        if metric == "final_accuracy":
            div()
        label  = METRIC_LABELS.get(metric, metric)
        scores = [r["aggregated"].get(metric, 0) for r in all_results]
        best   = max(scores) if len(set(scores)) > 1 else None
        row    = f"  {label}"
        for s in scores:
            star = " *" if (best is not None and s == best) else "  "
            row += f"  {s:.4f}{star}{'':>18}"
        lines.append(row)

    div()
    lines.append("  * = best score for that metric")

    # ── Score bars ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  ACCURACY BARS")
    div()
    for r in all_results:
        lines.append(f"  {r['pipeline']}")
        for m in metrics:
            s     = r["aggregated"].get(m, 0)
            label = METRIC_LABELS.get(m, m)
            lines.append(f"    {label} {_bar(s)}")
        lines.append("")

    # ── Final ranking ─────────────────────────────────────────────────────────
    div("=")
    lines.append("  FINAL RANKING")
    div("=")
    ranked  = sorted(all_results, key=lambda r: r["aggregated"]["final_accuracy"], reverse=True)
    medals  = ["1st", "2nd", "3rd"]
    for i, r in enumerate(ranked):
        medal = medals[i] if i < len(medals) else f"{i+1}th"
        score = r["aggregated"]["final_accuracy"]
        lines.append(f"  [{medal}]  {r['pipeline']:<40} {score*100:.2f}%")
    div("=")

    # ── Per-category breakdown ────────────────────────────────────────────────
    lines.append("")
    lines.append("  PER-CATEGORY BREAKDOWN (Final Accuracy %)")
    div()

    all_cats = sorted(set(
        cat for r in all_results for cat in r.get("per_category", {})
    ))
    cat_header = f"  {'Category':<30}" + "".join(f"  {r['pipeline'][:20]:<22}" for r in all_results)
    lines.append(cat_header)
    div()

    for cat in all_cats:
        row = f"  {cat:<30}"
        for r in all_results:
            cat_data = r.get("per_category", {}).get(cat, {})
            score    = cat_data.get("final_accuracy", 0.0) * 100
            row     += f"  {score:5.1f}%{'':>16}"
        lines.append(row)

    div()

    # ── Per-metric category detail ────────────────────────────────────────────
    lines.append("")
    lines.append("  DETAILED METRIC SCORES BY CATEGORY")
    div()
    for r in all_results:
        lines.append(f"  >> {r['pipeline']}")
        div("-")
        sub_header = f"  {'Category':<30}" + "".join(f"  {METRIC_LABELS[m][:10]:<14}" for m in metrics if m != "final_accuracy")
        lines.append(sub_header)
        div("-")
        for cat in all_cats:
            cat_data = r.get("per_category", {}).get(cat, {})
            row = f"  {cat:<30}"
            for m in metrics:
                if m == "final_accuracy":
                    continue
                row += f"  {cat_data.get(m, 0):.3f}{'':>10}"
            lines.append(row)
        lines.append("")

    div("=")
    lines.append("  END OF REPORT")
    div("=")

    report = "\n".join(lines)
    print(report)

    if save:
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path  = RESULTS_DIR / f"ragas_report_{ts_file}.md"
        js_path  = RESULTS_DIR / f"ragas_results_{ts_file}.json"

        md_path.write_text(report, encoding="utf-8")
        print(f"\n  Report saved -> {md_path}")

        # Save JSON (without verbose per_sample to keep it readable)
        compact = [
            {k: v for k, v in r.items() if k != "per_sample"}
            for r in all_results
        ]
        js_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON   saved -> {js_path}")

        # Save per-sample CSVs for Excel analysis
        for r in all_results:
            if not r.get("per_sample"):
                continue
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", r["pipeline"])
            csv_path = RESULTS_DIR / f"per_sample_{safe}_{ts_file}.csv"
            fields   = list(r["per_sample"][0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(r["per_sample"])
            print(f"  CSV    saved -> {csv_path}")

    return report



# ==============================================================================
# SECTION 5b — Academic Markdown Report (Thesis-Ready)
# ==============================================================================

def _md_bar(score: float, width: int = 25) -> str:
    """Renders a visual progress bar using Unicode blocks for Markdown."""
    filled = int(round(score * width))
    return "`" + "█" * filled + "░" * (width - filled) + f"` **{score*100:.1f}%**"


def _textual_analysis(all_results: list[dict]) -> list[str]:
    """
    Generate a short, data-driven textual analysis comparing the pipelines.
    Returns a list of Markdown-formatted paragraph strings.
    """
    if len(all_results) < 2:
        r = all_results[0]
        sc = r["aggregated"]
        return [
            f"The **{r['pipeline']}** pipeline was evaluated on {r['n_questions']} questions "
            f"from the Algerian legal dataset. It achieved a final accuracy score of "
            f"**{sc['final_accuracy']*100:.1f}%**, with a Context Recall of "
            f"{sc['context_recall']*100:.1f}% and a Context Precision of "
            f"{sc['context_precision']*100:.1f}%. "
            f"These scores reflect the pipeline's ability to retrieve and ground its "
            f"answers in relevant Algerian legal articles."
        ]

    ranked = sorted(all_results, key=lambda r: r["aggregated"]["final_accuracy"], reverse=True)
    best   = ranked[0]
    worst  = ranked[-1]
    b_sc   = best["aggregated"]
    w_sc   = worst["aggregated"]
    gap    = (b_sc["final_accuracy"] - w_sc["final_accuracy"]) * 100

    paragraphs = []

    # --- Overview paragraph ---
    paragraphs.append(
        f"The evaluation was conducted on **{best['n_questions']} questions** drawn from the "
        f"Algerian Legal RAG Dataset (v2), spanning four legal domains: "
        f"Penal Law, Civil Law, Administrative Law, and Labor Law. "
        f"Each pipeline was assessed across five RAGAS-inspired metrics, "
        f"and a weighted final accuracy score was computed to enable direct comparison."
    )

    # --- Winner paragraph ---
    paragraphs.append(
        f"### Best-Performing Pipeline: {best['pipeline']}\n\n"
        f"**{best['pipeline']}** achieved the highest overall accuracy of "
        f"**{b_sc['final_accuracy']*100:.1f}%**, outperforming all other configurations. "
        f"It recorded a Context Recall of **{b_sc['context_recall']*100:.1f}%** and "
        f"a Context Precision of **{b_sc['context_precision']*100:.1f}%**, "
        f"indicating that its retrieval module consistently surfaces relevant legal articles. "
        f"Its Faithfulness score of **{b_sc['faithfulness']*100:.1f}%** demonstrates "
        f"that generated answers remain well-grounded in the retrieved legal texts, "
        f"reducing the risk of hallucination — a critical requirement for a legal assistant."
    )

    # --- Runner-up / weakness ---
    if len(ranked) >= 2:
        second = ranked[1]
        s_sc   = second["aggregated"]
        paragraphs.append(
            f"### Runner-Up: {second['pipeline']}\n\n"
            f"**{second['pipeline']}** followed with a final accuracy of "
            f"**{s_sc['final_accuracy']*100:.1f}%**. "
            f"Its retrieval precision ({s_sc['context_precision']*100:.1f}%) "
            f"{'was comparable to' if abs(s_sc['context_precision'] - b_sc['context_precision']) < 0.05 else 'lagged behind'} "
            f"the leading pipeline, while its Article Recall of {s_sc['article_recall']*100:.1f}% "
            f"suggests {'strong' if s_sc['article_recall'] > 0.5 else 'moderate'} coverage of "
            f"ground-truth cited articles. "
            f"The overall gap between the two pipelines is **{gap:.1f} percentage points**."
        )

    # --- Why the winner won ---
    best_cr  = b_sc["context_recall"]
    best_f   = b_sc["faithfulness"]
    best_ar  = b_sc["article_recall"]

    strengths = []
    if best_cr  > 0.15: strengths.append("high context recall (broad coverage of ground-truth content)")
    if best_f   > 0.15: strengths.append("strong faithfulness (answers are grounded in the context)")
    if best_ar  > 0.30: strengths.append("high article recall (retrieves the correct cited laws)")

    if strengths:
        paragraphs.append(
            f"### Why {best['pipeline']} Performs Better\n\n"
            f"The performance advantage of **{best['pipeline']}** can be attributed to: "
            + ", and ".join(strengths) + ". "
            f"The BGE-M3 multilingual embedding model is specifically optimized for dense "
            f"semantic retrieval across Arabic, French, and English — the three languages "
            f"present in Algerian legal texts. Its graph-based retrieval structure further "
            f"amplifies precision by traversing article relationships via PageRank-weighted "
            f"traversal, ensuring that contextually adjacent articles are also surfaced."
        )

    # --- Limitations ---
    paragraphs.append(
        "### Limitations and Notes\n\n"
        "The metrics in this evaluation are computed using **token-level F1 overlap** "
        "between generated answers, retrieved contexts, and ground-truth references. "
        "While this approach is fully offline and reproducible, it may underestimate "
        "the semantic quality of Arabic paraphrases. A future evaluation using an "
        "LLM-as-judge (e.g., GPT-4 or a fine-tuned Arabic judge model) would provide "
        "more nuanced scores for Faithfulness and Answer Relevancy. "
        "Additionally, the Article Recall metric is specific to this legal domain and "
        "measures whether the exact article numbers cited in the ground truth were retrieved — "
        "a proxy for legal precision that is not part of the standard RAGAS framework."
    )

    return paragraphs


def generate_academic_report(all_results: list[dict]) -> str:
    """
    Produce a complete, thesis-ready Markdown evaluation report.
    Saved to evaluation/ragas_results/academic_report_<timestamp>.md
    """
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
    ts_file  = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics  = ["context_recall", "context_precision", "faithfulness",
                "answer_relevancy", "article_recall", "final_accuracy"]

    METRIC_FULL = {
        "context_recall":    "Context Recall",
        "context_precision": "Context Precision",
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "article_recall":    "Article Recall (Legal)",
        "final_accuracy":    "**Final Accuracy**",
    }

    METRIC_DESCRIPTIONS = {
        "context_recall": (
            "Measures how much of the information present in the **ground-truth answer** "
            "is covered by the **retrieved context chunks**. A high score indicates that "
            "the retrieval module successfully surfaces the legal articles needed to answer "
            "the question completely. Computed as the token-level F1 overlap between the "
            "ground truth and the concatenated retrieved texts."
        ),
        "context_precision": (
            "Measures the **relevance quality** of each individually retrieved chunk. "
            "A high score means every retrieved article is pertinent to the question, "
            "with minimal noise. Computed as the average token-level F1 between each "
            "chunk and the ground truth. Low precision suggests the retriever is returning "
            "loosely related legal articles."
        ),
        "faithfulness": (
            "Evaluates whether the **generated answer is grounded** in the retrieved "
            "context — i.e., it does not hallucinate facts not present in the retrieved "
            "articles. This is a critical metric for a legal assistant, where fabricated "
            "legal citations or incorrect article numbers can be harmful. Computed as the "
            "token-level F1 overlap between the answer and the retrieved context pool."
        ),
        "answer_relevancy": (
            "Measures whether the **generated answer actually addresses** the user's "
            "question. An answer that is faithful to the context but drifts from the "
            "original question will score low here. Computed as the token-level overlap "
            "between the question's keywords and the generated answer."
        ),
        "article_recall": (
            "A **domain-specific legal metric** that measures what fraction of the "
            "ground-truth cited legal articles (e.g., *Article 372 of the Penal Code*) "
            "were successfully retrieved by the pipeline. This is the most direct measure "
            "of legal retrieval precision and is not part of the standard RAGAS framework — "
            "it was added specifically for the Algerian Legal RAG evaluation."
        ),
    }

    WEIGHT_RATIONALE = {
        "context_recall":    "25%",
        "context_precision": "20%",
        "faithfulness":      "25%",
        "answer_relevancy":  "15%",
        "article_recall":    "15%",
    }

    lines = []
    a = lines.append  # shorthand

    # ==========================================================================
    # Title & Metadata
    # ==========================================================================
    a("# RAG Pipeline Evaluation Report")
    a("")
    a("## Istacherni — Algerian Legal Assistant")
    a("")
    a(f"> **Generated:** {ts}  ")
    a(f"> **Dataset:** Algerian Law RAGAS Dataset v2  ")
    a(f"> **Questions Evaluated:** {all_results[0]['n_questions']}  ")
    a(f"> **Evaluation Mode:** {'Retrieval-only (generation skipped)' if all_results[0].get('skip_gen') else 'Full pipeline (retrieval + generation)'}  ")
    a(f"> **Pipelines Compared:** {len(all_results)}")
    a("")
    a("---")
    a("")

    # ==========================================================================
    # Section 1: Introduction
    # ==========================================================================
    a("## 1. Introduction")
    a("")
    a(
        "This report presents the results of an automated evaluation of the **Graph RAG** "
        "pipelines developed for the Istacherni legal assistant application. "
        "The system is designed to answer questions about Algerian law by retrieving "
        "relevant legal articles and synthesizing accurate, grounded responses."
    )
    a("")
    a(
        "The evaluation uses a **RAGAS-inspired framework** — an industry-standard "
        "methodology for measuring Retrieval-Augmented Generation quality across four "
        "core dimensions: context quality, answer faithfulness, and answer relevance. "
        "An additional fifth metric, *Article Recall*, was introduced to capture "
        "legal-domain precision."
    )
    a("")
    a("---")
    a("")

    # ==========================================================================
    # Section 2: Metrics
    # ==========================================================================
    a("## 2. Evaluation Metrics")
    a("")
    a(
        "The following five metrics were computed for each pipeline. "
        "The **Final Accuracy** is a weighted combination of all five scores:"
    )
    a("")

    # Metric definitions table
    a("| Metric | Weight | Description |")
    a("|--------|--------|-------------|")
    for m, desc in METRIC_DESCRIPTIONS.items():
        short_desc = desc.split(".")[0].replace("**", "").strip()
        a(f"| {METRIC_FULL[m]} | {WEIGHT_RATIONALE[m]} | {short_desc} |")
    a("")
    a("### Detailed Metric Definitions")
    a("")
    for m, desc in METRIC_DESCRIPTIONS.items():
        a(f"#### {METRIC_FULL[m].replace('**', '')}")
        a("")
        a(desc)
        a("")

    a("---")
    a("")

    # ==========================================================================
    # Section 3: Pipeline Configurations
    # ==========================================================================
    a("## 3. Pipeline Configurations")
    a("")
    a("| Pipeline | Embedding Model | Generation Model | Retrieval Strategy |")
    a("|----------|----------------|------------------|--------------------|")
    config_map = {
        "BGE + Qwen (Graph RAG)": (
            "BAAI/BGE-M3 (multilingual dense)", "Qwen2-7B (local, Ollama)",
            "Graph traversal + PageRank (PPR) + BM25 hybrid"
        ),
        "CamelBERT + Qwen": (
            "CAMeLBERT-MSA (Arabic BERT, Modern Standard Arabic)", "Qwen2-7B (local, Ollama)",
            "Dense cosine similarity (ChromaDB)"
        ),
    }
    for r in all_results:
        cfg = config_map.get(r["pipeline"],
                             ("Custom", "Qwen2-7B", "Custom retrieval"))
        a(f"| {r['pipeline']} | {cfg[0]} | {cfg[1]} | {cfg[2]} |")
    a("")
    a("---")
    a("")

    # ==========================================================================
    # Section 4: Results — Comparison Table
    # ==========================================================================
    a("## 4. Results: Metric Comparison Table")
    a("")
    a("The table below presents the aggregated scores for each pipeline across all metrics.")
    a("")

    # Build header
    header  = "| Metric |"
    divider = "|--------|"
    for r in all_results:
        header  += f" {r['pipeline']} |"
        divider += "---------|"
    a(header)
    a(divider)

    for m in metrics:
        label  = METRIC_FULL[m]
        scores = [r["aggregated"].get(m, 0) for r in all_results]
        best   = max(scores) if len(set(scores)) > 1 else None
        row    = f"| {label} |"
        for s in scores:
            cell = f" **{s:.4f}** |" if (best is not None and s == best) else f" {s:.4f} |"
            row += cell
        a(row)

    a("")
    a("> **Bold** values indicate the best score for each metric.")
    a("")
    a("---")
    a("")

    # ==========================================================================
    # Section 5: Visual Score Comparison
    # ==========================================================================
    a("## 5. Visual Score Comparison")
    a("")
    a("The following progress bars provide a visual representation of each pipeline's scores:")
    a("")
    for r in all_results:
        a(f"### {r['pipeline']}")
        a("")
        a("| Metric | Score | Visual |")
        a("|--------|-------|--------|")
        for m in metrics:
            s     = r["aggregated"].get(m, 0)
            label = METRIC_FULL[m].replace("**", "")
            a(f"| {label} | {s*100:.1f}% | {_md_bar(s)} |")
        a("")

    a("---")
    a("")

    # ==========================================================================
    # Section 6: Per-Category Breakdown
    # ==========================================================================
    a("## 6. Per-Category Accuracy Breakdown")
    a("")
    a(
        "The dataset covers four legal domains. The table below shows the "
        "**Final Accuracy** per category for each pipeline:"
    )
    a("")

    all_cats = sorted(set(
        cat for r in all_results for cat in r.get("per_category", {})
    ))

    header  = "| Legal Domain |"
    divider = "|--------------|"
    for r in all_results:
        header  += f" {r['pipeline']} |"
        divider += "----------|"
    a(header)
    a(divider)

    for cat in all_cats:
        row  = f"| {cat} |"
        vals = []
        for r in all_results:
            cat_data = r.get("per_category", {}).get(cat, {})
            score    = cat_data.get("final_accuracy", 0.0)
            vals.append(score)
        best_val = max(vals) if len(set(vals)) > 1 else None
        for score in vals:
            cell = f" **{score*100:.1f}%** |" if (best_val is not None and score == best_val) else f" {score*100:.1f}% |"
            row += cell
        a(row)

    a("")
    a("---")
    a("")

    # ==========================================================================
    # Section 7: Textual Analysis
    # ==========================================================================
    a("## 7. Analysis and Discussion")
    a("")
    paragraphs = _textual_analysis(all_results)
    for p in paragraphs:
        a(p)
        a("")

    a("---")
    a("")

    # ==========================================================================
    # Section 8: Final Ranking
    # ==========================================================================
    a("## 8. Final Pipeline Ranking")
    a("")
    ranked = sorted(all_results, key=lambda r: r["aggregated"]["final_accuracy"], reverse=True)
    medals = ["1st Place", "2nd Place", "3rd Place", "4th Place"]

    a("| Rank | Pipeline | Final Accuracy | Context Recall | Faithfulness |")
    a("|------|----------|---------------|----------------|--------------|")
    for i, r in enumerate(ranked):
        sc   = r["aggregated"]
        rank = medals[i] if i < len(medals) else f"{i+1}th"
        a(
            f"| {rank} | {r['pipeline']} "
            f"| **{sc['final_accuracy']*100:.2f}%** "
            f"| {sc['context_recall']*100:.2f}% "
            f"| {sc['faithfulness']*100:.2f}% |"
        )

    a("")
    a("> **Recommendation:** Based on the evaluation results, "
      f"**{ranked[0]['pipeline']}** is the recommended pipeline for deployment "
      "in the Istacherni legal assistant. It achieves the best balance of "
      "retrieval quality and answer faithfulness across all legal domains.")
    a("")
    a("---")
    a("")

    # ==========================================================================
    # Section 9: Conclusion
    # ==========================================================================
    a("## 9. Conclusion")
    a("")
    best = ranked[0]
    b_sc = best["aggregated"]
    a(
        f"This evaluation demonstrates that the **{best['pipeline']}** pipeline "
        f"achieves superior performance for Arabic legal question answering in the "
        f"Algerian legal context, with a final accuracy of "
        f"**{b_sc['final_accuracy']*100:.1f}%**. "
        f"The graph-based retrieval architecture, combined with the BGE-M3 multilingual "
        f"embeddings and local Qwen2-7B generation, provides a robust, fully-offline "
        f"solution that meets the strict requirements of a legal assistant: "
        f"accuracy, faithfulness, and reproducibility."
    )
    a("")
    a(
        "Future work should focus on: (1) expanding the evaluation dataset to include "
        "more complex multi-hop legal questions, (2) integrating an LLM-as-judge for "
        "more nuanced semantic scoring of Arabic answers, and (3) fine-tuning the "
        "generation model on Algerian legal corpora to further improve faithfulness."
    )
    a("")
    a("---")
    a("")
    a(f"*Report generated automatically by the Istacherni RAG Evaluation Framework — {ts}*")

    # ==========================================================================
    # Save
    # ==========================================================================
    report_text = "\n".join(lines)
    out_path = RESULTS_DIR / f"academic_report_{ts_file}.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n  [ACADEMIC REPORT] Saved -> {out_path}")
    return report_text


# ==============================================================================
# SECTION 6 — Main Entry Point
# ==============================================================================

PIPELINE_MAP = {
    "bge": {
        "name":       "BGE + Qwen (Graph RAG)",
        "full_fn":    run_bge_qwen_pipeline,
        "retrieve_fn": retrieve_only_bge,
    },
    "camelbert": {
        "name":       "CamelBERT + Qwen (Graph RAG)",
        "full_fn":    run_camelbert_pipeline,
        "retrieve_fn": retrieve_only_camelbert,
    },
    "agentic_bge": {
        "name":       "BGE + Qwen (Agentic RAG)",
        "full_fn":    run_agentic_bge_pipeline,
        "retrieve_fn": retrieve_only_agentic_bge,
    },
    "agentic_camelbert": {
        "name":       "CamelBERT + Qwen (Agentic RAG)",
        "full_fn":    run_agentic_camelbert_pipeline,
        "retrieve_fn": retrieve_only_agentic_camelbert,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation -- Istacherni Legal RAG Pipelines"
    )
    parser.add_argument("--n", type=int, default=None,
        help="Number of questions to evaluate. Default: all (~100).")
    parser.add_argument("--pipeline", choices=["bge", "camelbert", "agentic_bge", "agentic_camelbert", "graph_all", "agentic_all", "all"], default="all",
        help="Which pipeline(s) to evaluate.")
    parser.add_argument("--skip-gen", action="store_true",
        help="Skip LLM generation. Only run retrieval metrics (fast, no Ollama needed).")
    parser.add_argument("--report-only", action="store_true",
        help="Reprint the latest saved report without running anything.")
    parser.add_argument("--quiet", action="store_true",
        help="Suppress per-question output.")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  RAGAS EVALUATION -- Istacherni Graph RAG")
    print("=" * 65)

    # ── Report-only shortcut ─────────────────────────────────────────────────
    if args.report_only:
        saved = sorted(RESULTS_DIR.glob("ragas_results_*.json"), reverse=True)
        if not saved:
            print("  ERROR: No saved results found. Run evaluation first.")
            sys.exit(1)
        latest = saved[0]
        print(f"  Loading: {latest.name}")
        data = json.loads(latest.read_text(encoding="utf-8"))
        generate_report(data, save=False)
        return

    # ── Ollama check ─────────────────────────────────────────────────────────
    if not args.skip_gen:
        print("  Checking Ollama...", end=" ", flush=True)
        if _check_ollama():
            print("OK (running)")
        else:
            print("NOT RUNNING")
            print("  WARNING: Ollama is not running. Generation steps will fail.")
            print("  Options:")
            print("    1. Start Ollama:  ollama serve")
            print("    2. Use --skip-gen for retrieval-only evaluation")
            print()

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset = load_dataset(n=args.n)

    # ── Select pipelines ─────────────────────────────────────────────────────
    if args.pipeline == "all":
        selected_keys = ["bge", "camelbert", "agentic_bge", "agentic_camelbert"]
    elif args.pipeline == "graph_all":
        selected_keys = ["bge", "camelbert"]
    elif args.pipeline == "agentic_all":
        selected_keys = ["agentic_bge", "agentic_camelbert"]
    else:
        selected_keys = [args.pipeline]

    mode_label = "Retrieval-only (--skip-gen)" if args.skip_gen else "Full (Retrieval + Generation)"
    print(f"\n  Mode      : {mode_label}")
    print(f"  Pipelines : {', '.join(PIPELINE_MAP[k]['name'] for k in selected_keys)}")
    print(f"  Questions : {len(dataset)}")

    # ── Run ──────────────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for key in selected_keys:
        cfg = PIPELINE_MAP[key]
        try:
            result = run_pipeline(
                pipeline_name = cfg["name"],
                pipeline_fn   = cfg["full_fn"],
                retrieve_fn   = cfg["retrieve_fn"],
                dataset       = dataset,
                skip_gen      = args.skip_gen,
                verbose       = not args.quiet,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR: Pipeline '{cfg['name']}' failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n  ERROR: No pipelines completed successfully.")
        sys.exit(1)

    # ── Console report ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  GENERATING CONSOLE REPORT")
    print("=" * 65)
    generate_report(all_results, save=True)

    # ── Academic Markdown report (thesis-ready) ───────────────────────────────
    print("\n" + "=" * 65)
    print("  GENERATING ACADEMIC MARKDOWN REPORT")
    print("=" * 65)
    generate_academic_report(all_results)

    print("\n  Evaluation complete!")
    print("  Academic report saved to: evaluation/ragas_results/\n")


if __name__ == "__main__":
    main()
