# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output on Windows to avoid cp1252 errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
=============================================================
 RAGAS-Style Evaluation Script for Algerian Legal RAG Models
=============================================================
Compares 4 embedding models (BGE-M3, MiniLM, Qwen, CamELBERT)
using RAGAS-inspired metrics computed WITHOUT an LLM judge:
  - Answer Similarity   (sentence embedding cosine similarity)
  - Answer Correctness  (weighted F1 over key legal terms)
  - Faithfulness Proxy  (fraction of answer that appears in ground_truth)
  - Context Recall Proxy (key articles overlap)

Outputs:
  - Per-question scores in CSV
  - Summary table in terminal
  - Final radar chart (PNG)
"""

import json
import re
import os
import sys
import time
import unicodedata
import collections
from pathlib import Path

# -- Optional rich progress bar -----------------------------------------
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# -- Sentence-transformer for semantic similarity ------------------------
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARN] sentence-transformers not found -> Answer Similarity will use token overlap only.")

# -- matplotlib for radar chart ------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("[WARN] matplotlib/numpy not found -> radar chart will be skipped.")


# ========================================================================
# CONFIG - Edit answer file paths if needed
# ========================================================================
BASE_DIR = Path(__file__).parent

MODELS = {
    "BGE-M3":    BASE_DIR / "answers_bge_bm25_final_1.txt",
    "MiniLM":    BASE_DIR / "answers_minilm_bm25_final.txt",
    "Qwen":      BASE_DIR / "answers_qwen_bm25_final.txt",
    "CamELBERT": BASE_DIR / "answers_camelbert_bm25_final.txt",
}

DATASET_PATH = BASE_DIR / "algerian_law_ragas_dataset_v3.json"
OUTPUT_CSV   = BASE_DIR / "ragas_evaluation_results.csv"
OUTPUT_CHART = BASE_DIR / "ragas_radar_chart.png"

# Embedding model for semantic similarity (small, fast, multilingual)
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Weights for the final Accuracy score
WEIGHTS = {
    "answer_similarity":  0.35,
    "answer_correctness": 0.30,
    "faithfulness":       0.20,
    "article_recall":     0.15,
}

SEPARATOR = "=" * 50  # separator between Q&A blocks in answer files


# ========================================================================
# TEXT UTILITIES
# ========================================================================

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics, unify chars, lowercase."""
    # Remove diacritics (harakat)
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    # Normalize alef variants
    text = re.sub(r'[آأإ]', 'ا', text)
    # Normalize teh marbuta
    text = text.replace('ة', 'ه')
    # Normalize yeh variants
    text = text.replace('ى', 'ي')
    # Normalize waw with hamza
    text = text.replace('ؤ', 'و')
    text = text.replace('ئ', 'ي')
    # Strip punctuation and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Normalize and split text into tokens."""
    return normalize_arabic(text).split()


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = tokenize(prediction)
    ref_tokens  = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = collections.Counter(pred_tokens)
    ref_counter  = collections.Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())
    precision = common / len(pred_tokens)
    recall    = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_overlap_ratio(answer: str, reference: str) -> float:
    """What fraction of answer tokens appear in reference (faithfulness proxy)."""
    pred_tokens = tokenize(answer)
    ref_set = set(tokenize(reference))
    if not pred_tokens:
        return 0.0
    covered = sum(1 for t in pred_tokens if t in ref_set)
    return covered / len(pred_tokens)


def extract_articles_from_text(text: str) -> set[str]:
    """
    Extract legal article references from a text string.
    Patterns like: المادة 372, المواد 350-359, م 376, مادة 214
    """
    articles = set()
    # Arabic: المادة NNN, المواد NNN-NNN
    patterns = [
        r'المادة\s+(\d+(?:\s*مكرر\s*\d*)?)',
        r'المواد\s+(\d+(?:\s*[--]\s*\d+)?)',
        r'م\.?\s*(\d+)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            num = re.sub(r'\s+', '', m.group(1))
            articles.add(num)
    return articles


def article_recall(predicted_articles: set, ground_truth_articles: list) -> float:
    """What fraction of ground-truth articles are mentioned in the prediction."""
    gt_nums: set[str] = set()
    for art in ground_truth_articles:
        # Extract numbers specifically associated with articles
        nums = re.findall(r'(?:المادة|المواد|م|مادة)\s*(\d+)', art)
        if not nums:
            # Fallback for strings that are just numbers
            nums = re.findall(r'^\s*(\d+)\s*$', art)
        if not nums:
            # Last resort fallback (avoiding years like 1966, 2024 if possible)
            all_nums = re.findall(r'\d+', art)
            nums = [n for n in all_nums if len(n) < 4] # very crude heuristic to skip years
        
        gt_nums.update(nums)
    
    if not gt_nums:
        return 1.0  # nothing to recall
        
    pred_nums = set(re.findall(r'\d+', ' '.join(predicted_articles)))
    found = gt_nums & pred_nums
    return len(found) / len(gt_nums)


# ========================================================================
# DATASET LOADER
# ========================================================================

def load_dataset(path: Path) -> list[dict]:
    """Load and validate the RAGAS JSON dataset."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} questions from dataset.")
    return data


# ========================================================================
# ANSWER FILE PARSER
# ========================================================================

def parse_answer_file(path: Path) -> dict[str, dict]:
    """
    Parse an answer .txt file and return a dict keyed by normalized question.
    Each value contains 'answer' and 'sources'.
    Format per block:
        [User]: <question>
        ANSWER:
        <answer text>
        (Time taken: N seconds)
        SOURCES:
        [1] ...
        ==================================================
    """
    results: dict[str, dict] = {}
    try:
        raw = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        print(f"[WARN] Answer file not found: {path}")
        return results

    blocks = re.split(r'={40,}', raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract question
        q_match = re.search(r'\[User\]:\s*(.+?)(?:\r?\n)', block)
        if not q_match:
            continue
        question = q_match.group(1).strip()

        # Extract answer (between ANSWER: and SOURCES: or end)
        ans_match = re.search(r'ANSWER:\r?\n(.*?)(?:SOURCES:|$)', block, re.DOTALL)
        if not ans_match:
            # Try without SOURCES
            ans_match = re.search(r'ANSWER:\r?\n(.*)', block, re.DOTALL)
        answer = ans_match.group(1).strip() if ans_match else ""

        # Remove timing line from answer
        answer = re.sub(r'\(Time taken:.*?\)', '', answer).strip()

        # Extract sources
        sources_match = re.search(r'SOURCES:\r?\n(.*)', block, re.DOTALL)
        sources_text = sources_match.group(1).strip() if sources_match else ""

        norm_q = normalize_arabic(question)
        results[norm_q] = {
            "question": question,
            "answer": answer,
            "sources_text": sources_text,
        }

    print(f"  -> Parsed {len(results)} answers from {path.name}")
    return results


# ========================================================================
# SBERT LOADER (singleton)
# ========================================================================
_sbert_model = None

def get_sbert():
    global _sbert_model
    if _sbert_model is None and SBERT_AVAILABLE:
        print(f"[INFO] Loading SBERT model: {SBERT_MODEL_NAME} ...")
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        print("[INFO] SBERT model loaded.")
    return _sbert_model


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using SBERT (multilingual)."""
    model = get_sbert()
    if model is None:
        # Fallback to token F1
        return token_f1(text_a, text_b)
    emb_a = model.encode(text_a, convert_to_tensor=True, show_progress_bar=False)
    emb_b = model.encode(text_b, convert_to_tensor=True, show_progress_bar=False)
    score = float(st_util.cos_sim(emb_a, emb_b))
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ========================================================================
# EVALUATION ENGINE
# ========================================================================

def evaluate_model(model_name: str,
                   answers_map: dict,
                   dataset: list[dict]) -> list[dict]:
    """
    Evaluate a single model against the dataset.
    Returns a list of per-question score dicts.
    """
    records = []
    iterator = tqdm(dataset, desc=f"Evaluating {model_name}", unit="q") if USE_TQDM else dataset

    for item in iterator:
        question    = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        gt_articles  = item.get("articles", [])

        norm_q = normalize_arabic(question)

        # Find best matching answer (fuzzy if exact key not found)
        entry = answers_map.get(norm_q)
        if entry is None:
            # Try partial match (longest common key)
            best_key = None
            best_len = 0
            q_tokens = set(tokenize(question))
            for key in answers_map:
                overlap = len(q_tokens & set(key.split()))
                if overlap > best_len:
                    best_len = overlap
                    best_key = key
            if best_key and best_len >= max(2, len(q_tokens) // 2):
                entry = answers_map[best_key]

        if entry is None:
            # No answer found for this question
            records.append({
                "model":             model_name,
                "question":          question,
                "found":             False,
                "answer_similarity": 0.0,
                "answer_correctness":0.0,
                "faithfulness":      0.0,
                "article_recall":    0.0,
                "accuracy":          0.0,
            })
            continue

        answer = entry["answer"]

        # 1. Answer Similarity (semantic)
        sim_score = semantic_similarity(answer, ground_truth)

        # 2. Answer Correctness (token F1)
        f1_score = token_f1(answer, ground_truth)

        # 3. Faithfulness proxy (how much of the answer is grounded in GT)
        faith_score = token_overlap_ratio(answer, ground_truth)

        # 4. Article Recall
        pred_articles = extract_articles_from_text(answer + " " + entry["sources_text"])
        art_rec = article_recall(pred_articles, gt_articles)

        # Weighted accuracy
        accuracy = (
            WEIGHTS["answer_similarity"]  * sim_score  +
            WEIGHTS["answer_correctness"] * f1_score   +
            WEIGHTS["faithfulness"]       * faith_score +
            WEIGHTS["article_recall"]     * art_rec
        )

        records.append({
            "model":             model_name,
            "question":          question,
            "found":             True,
            "answer_similarity": round(sim_score,  4),
            "answer_correctness":round(f1_score,   4),
            "faithfulness":      round(faith_score, 4),
            "article_recall":    round(art_rec,    4),
            "accuracy":          round(accuracy,   4),
        })

    return records


# ========================================================================
# REPORTING
# ========================================================================

def write_csv(all_records: list[dict], path: Path):
    """Write per-question scores to CSV."""
    import csv
    fieldnames = [
        "model", "question", "found",
        "answer_similarity", "answer_correctness",
        "faithfulness", "article_recall", "accuracy"
    ]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)
    print(f"\n[INFO] Per-question results saved -> {path}")


def compute_summary(all_records: list[dict]) -> dict[str, dict]:
    """Compute average scores per model."""
    from collections import defaultdict
    sums   = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    metrics = ["answer_similarity", "answer_correctness", "faithfulness",
               "article_recall", "accuracy"]
    for rec in all_records:
        model = rec["model"]
        if rec["found"]:
            for m in metrics:
                sums[model][m] += rec[m]
            counts[model] += 1
    summary = {}
    for model in sums:
        total = counts[model]
        summary[model] = {m: round(sums[model][m] / total, 4) if total else 0.0
                          for m in metrics}
        summary[model]["answered"] = total
    return summary


def print_summary_table(summary: dict[str, dict], dataset_size: int):
    """Pretty-print the summary table."""
    col_w = 16
    metrics = [
        ("Similarity",   "answer_similarity"),
        ("Correctness",  "answer_correctness"),
        ("Faithfulness", "faithfulness"),
        ("Art. Recall",  "article_recall"),
        ("Accuracy",     "accuracy"),
    ]
    header = f"{'Model':<14}" + "".join(f"{label:>{col_w}}" for label, _ in metrics) + f"{'Answered':>{col_w}}"
    print("\n" + "=" * len(header))
    print("  RAGAS-Style Evaluation Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Sort by accuracy descending
    ranked = sorted(summary.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
    for rank, (model, scores) in enumerate(ranked, 1):
        row = f"{rank}. {model:<12}"
        for _, key in metrics:
            val = scores.get(key, 0.0)
            row += f"{val:>{col_w}.4f}"
        row += f"{scores['answered']:>{col_w}}/{dataset_size}"
        print(row)

    print("-" * len(header))
    print(f"\nWeights: Similarity x{WEIGHTS['answer_similarity']}"
          f"  Correctness x{WEIGHTS['answer_correctness']}"
          f"  Faithfulness x{WEIGHTS['faithfulness']}"
          f"  Art.Recall x{WEIGHTS['article_recall']}")


def draw_radar_chart(summary: dict[str, dict], output_path: Path):
    """Draw a radar / spider chart for all models."""
    if not PLOT_AVAILABLE:
        return

    categories = ["Similarity", "Correctness", "Faithfulness", "Art. Recall", "Accuracy"]
    keys       = ["answer_similarity", "answer_correctness", "faithfulness",
                  "article_recall", "accuracy"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f23')

    for i, (model, scores) in enumerate(summary.items()):
        values = [scores.get(k, 0) for k in keys]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, color='white')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', fontsize=8)
    ax.tick_params(axis='x', pad=15)
    ax.grid(color='grey', linestyle='--', alpha=0.4)
    ax.spines['polar'].set_color('grey')

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                       fontsize=11, facecolor='#1a1a2e',
                       edgecolor='grey', labelcolor='white')

    ax.set_title("RAGAS-Style Model Accuracy Comparison\nAlgerian Legal RAG Pipeline",
                 fontsize=14, color='white', pad=25, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Radar chart saved -> {output_path}")


# ========================================================================
# MAIN
# ========================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  RAGAS Evaluation - Algerian Legal RAG Models")
    print("=" * 60)

    # 1. Load dataset
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)
    dataset = load_dataset(DATASET_PATH)

    # 2. Pre-load SBERT once
    if SBERT_AVAILABLE:
        get_sbert()

    # 3. Evaluate each model
    all_records = []
    for model_name, ans_path in MODELS.items():
        print(f"\n" + "-" * 50)
        print(f"[MODEL] {model_name}  ->  {ans_path.name}")
        if not ans_path.exists():
            print(f"  [SKIP] File not found: {ans_path}")
            continue
        answers_map = parse_answer_file(ans_path)
        records = evaluate_model(model_name, answers_map, dataset)
        all_records.extend(records)
        found = sum(1 for r in records if r["found"])
        avg_acc = sum(r["accuracy"] for r in records if r["found"]) / max(found, 1)
        print(f"  Matched {found}/{len(dataset)} questions | Avg Accuracy: {avg_acc:.4f}")

    if not all_records:
        print("[ERROR] No records generated. Check file paths.")
        sys.exit(1)

    # 4. Write CSV
    write_csv(all_records, OUTPUT_CSV)

    # 5. Compute & print summary
    summary = compute_summary(all_records)
    print_summary_table(summary, len(dataset))

    # 6. Radar chart
    draw_radar_chart(summary, OUTPUT_CHART)

    print(f"\n[DONE] Total time: {time.time() - t0:.1f}s")
    print(f"  Results CSV : {OUTPUT_CSV}")
    print(f"  Radar Chart : {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
