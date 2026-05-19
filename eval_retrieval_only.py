"""
Fast retrieval-only evaluation — no LLM judge.
Computes Precision, Recall, Hit Rate, MRR for all three RAG models.
"""
import json, re, unicodedata, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DATASET_PATH = "algerian_law_ragas_dataset_v3.json"
MODELS = {
    "Qwen RAG":       "answers_qwen_rag.txt",
    "Graph RAG":      "answers_graph_rag.txt",
    "CamELBERT RAG":  "answers_camelbert_rag.txt",
}

# ── Normalization ─────────────────────────────────────────────────────────────
_DIAC = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0640\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
)

# Arabic letter-variant map: normalise alef variants, tah marbuta, alef maqsura
# These variants appear in law names across different datasets and GT sources:
#   'قانون العمل الجزاىري' (ى) vs 'قانون العمل الجزائري' (ي) — silent match failure
_ARABIC_VARIANTS = str.maketrans({
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0622': '\u0627',  # آ → ا
    '\u0649': '\u064A',  # ى → ي  ← the key fix for 'جزاىري'
})

def normalize(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = _DIAC.sub('', s)
    s = s.translate(_ARABIC_VARIANTS)  # letter variant normalization
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_file(path: str) -> dict:
    """Returns {normalized_question: [normalized_source, ...]}"""
    if not Path(path).exists():
        print(f"  [WARN] File not found: {path}")
        return {}

    content = Path(path).read_text(encoding='utf-8')
    blocks  = content.split("==================================================")
    results = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = re.search(r'\[User\]:\s*(.*)', block)
        if not q_match:
            continue
        question = normalize(q_match.group(1).strip())

        sources_start = block.find("SOURCES:")
        if sources_start == -1:
            continue
        sources_raw = block[sources_start + len("SOURCES:"):].strip()

        seen   = set()
        srcs   = []
        for line in sources_raw.split('\n'):
            line = line.strip()
            if not line or not line.startswith('['):
                continue
            # Strip [N] index
            cleaned = re.sub(r'^\[\d+\]\s*', '', line)
            # Strip ALL parenthetical metadata: (score=...), (graph_score=..., pagerank=...), etc.
            cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned).strip()
            if not cleaned:
                continue
            norm_src = normalize(cleaned)
            if norm_src and norm_src not in seen:
                seen.add(norm_src)
                srcs.append(norm_src)

        results[question] = srcs

    return results

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(retrieved: list, ground_truth: list) -> dict:
    gt_set = set(ground_truth)   # already normalized + deduped

    hits = 0
    mrr  = 0.0
    for i, src in enumerate(retrieved):
        if src in gt_set:
            hits += 1
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)

    n_ret = len(retrieved)
    n_gt  = len(ground_truth)

    return {
        "Precision": hits / n_ret if n_ret else 0.0,
        "Recall":    hits / n_gt  if n_gt  else 0.0,
        "Hit Rate":  1.0 if hits > 0 else 0.0,
        "MRR":       mrr,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dataset = json.loads(Path(DATASET_PATH).read_text(encoding='utf-8'))
    print(f"Dataset: {len(dataset)} questions\n")

    # Build GT index: normalized_question → [normalized_articles]
    gt_index = {}
    for item in dataset:
        q   = normalize(item.get("question", "").strip())
        arts = list(dict.fromkeys(normalize(a) for a in item.get("articles", [])))
        gt_index[q] = arts

    summary = {}

    for model_name, path in MODELS.items():
        print(f"Evaluating {model_name}  ({path})")
        model_results = parse_file(path)

        agg = {"Precision": 0.0, "Recall": 0.0, "Hit Rate": 0.0, "MRR": 0.0}
        matched = count = 0

        for q_norm, gt_arts in gt_index.items():
            srcs = model_results.get(q_norm)
            if srcs is None:
                continue
            count += 1
            m = compute_metrics(srcs, gt_arts)
            for k in agg:
                agg[k] += m[k]
            if m["Hit Rate"] > 0:
                matched += 1

        if count > 0:
            for k in agg:
                agg[k] /= count
            summary[model_name] = agg
            print(f"  Questions evaluated: {count}  |  Questions with ≥1 hit: {matched}")
            print(f"  Precision={agg['Precision']:.4f}  Recall={agg['Recall']:.4f}  "
                  f"Hit Rate={agg['Hit Rate']:.4f}  MRR={agg['MRR']:.4f}")
        else:
            print("  No matching questions found.")
        print()

    # ── Print results table ───────────────────────────────────────────────────
    print("=" * 72)
    print("FINAL EVALUATION RESULTS  (retrieval metrics)")
    print("=" * 72)
    header = f"{'Model':<18} {'Precision':>10} {'Recall':>10} {'Hit Rate':>10} {'MRR':>10}"
    print(header)
    print("-" * 72)
    for model_name, m in summary.items():
        print(f"{model_name:<18} {m['Precision']:>10.4f} {m['Recall']:>10.4f} "
              f"{m['Hit Rate']:>10.4f} {m['MRR']:>10.4f}")
    print("=" * 72)

    # ── Save outputs ──────────────────────────────────────────────────────────
    md_lines = [
        "# RAG Retrieval Evaluation Results",
        "",
        f"**Dataset:** {len(dataset)} questions",
        "",
        "| Model | Precision | Recall | Hit Rate | MRR |",
        "|-------|----------:|-------:|---------:|----:|",
    ]
    for model_name, m in summary.items():
        md_lines.append(
            f"| {model_name} | {m['Precision']:.4f} | {m['Recall']:.4f} "
            f"| {m['Hit Rate']:.4f} | {m['MRR']:.4f} |"
        )
    md_lines += [
        "",
        "## Changes vs. Previous Run",
        "",
        "| Fix | Impact |",
        "|-----|--------|",
        "| Broad `\\s*\\([^)]*\\)` regex strips Graph RAG's `(graph_score=..., pagerank=...)` annotations | Graph RAG sources now parsed correctly |",
        "| Normalized question lookup (NFKC + diacritic strip) | Questions matched even with minor encoding differences |",
        "| Deduplication of retrieved sources before metric computation | Precision no longer deflated by duplicate retrievals |",
        "| Unicode NFKC + Arabic diacritic normalization on both sides | Robust cross-encoding source comparison |",
    ]

    Path("evaluation_retrieval_results.md").write_text('\n'.join(md_lines), encoding='utf-8')
    Path("evaluation_retrieval_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print("\nResults saved to evaluation_retrieval_results.md / .json")

if __name__ == "__main__":
    main()
