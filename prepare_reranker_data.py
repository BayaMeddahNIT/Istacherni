#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_reranker_data.py  —  Script 1 / 4
==========================================
Converts JSONL triplet files into flat (query, document, label) pairs
for cross-encoder reranker fine-tuning.

Oversampling (at TRIPLET level, before pair expansion):
  hard_negatives_failures.jsonl  -> 5x  (most critical)
  hard_negatives_new_laws.jsonl  -> 3x
  hard_negatives_relaxed.jsonl   -> 2x
  hard_negatives.jsonl           -> 1x

NOTE: Do NOT add triplets with delta > 0.15. Only the four
      existing files are used as-is.

Train/val split: 90 / 10 at QUERY level (no query spans both sets).

Outputs:
  reranker_train_pairs.json
  reranker_val_pairs.json
  data_stats.json
  reranker_triplet_eval.json   <- used by TripletEvaluator in Script 2
"""

import json
import random
import logging
import sys
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("prepare_reranker_data.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Source files and oversampling weights ─────────────────────────────────────
SOURCES = [
    ("hard_negatives_failures", PROJECT_ROOT / "hard_negatives_failures.jsonl", 5),
    ("hard_negatives_new_laws", PROJECT_ROOT / "hard_negatives_new_laws.jsonl", 3),
    ("hard_negatives_relaxed",  PROJECT_ROOT / "hard_negatives_relaxed.jsonl",  2),
    ("hard_negatives_main",     PROJECT_ROOT / "hard_negatives.jsonl",           1),
]

TRAIN_OUT    = PROJECT_ROOT / "reranker_train_pairs.json"
VAL_OUT      = PROJECT_ROOT / "reranker_val_pairs.json"
STATS_OUT    = PROJECT_ROOT / "data_stats.json"
TRIPLET_EVAL = PROJECT_ROOT / "reranker_triplet_eval.json"
SEED         = 42
VAL_RATIO    = 0.10


# ── Law code category detection ───────────────────────────────────────────────
def get_law_category(doc_text: str) -> str:
    """Detect law category from bracketed prefix '[قانون X - المادة N]'."""
    snippet = doc_text[:200]
    if "قانون العقوبات" in snippet:
        return "penal"
    if "القانون المدني" in snippet:
        return "civil"
    if "القانون التجاري" in snippet:
        return "commercial"
    if "90-11" in snippet or "قانون العمل" in snippet:
        return "labour"
    if "الإجراءات" in snippet or "الاجراءات" in snippet:
        return "procedure"
    return snippet[:20].strip()


# ── Load and validate JSONL ───────────────────────────────────────────────────
def load_and_validate_jsonl(path: Path, source_name: str) -> list:
    if not path.exists():
        log.warning("File not found (skipping): %s", path.name)
        return []
    valid, skipped = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("JSON error in %s line %d: %s", path.name, lineno, e)
                skipped += 1
                continue

            query = item.get("query", "")
            pos   = item.get("pos", [])
            neg   = item.get("neg", [])

            if not isinstance(query, str) or not query.strip():
                skipped += 1; continue
            if not isinstance(pos, list) or len(pos) != 1:
                log.warning("  [%s L%d] pos must have exactly 1 element, got %s",
                            path.name, lineno, len(pos) if isinstance(pos, list) else "?")
                skipped += 1; continue
            if not pos[0] or not isinstance(pos[0], str) or not pos[0].strip():
                skipped += 1; continue
            if not isinstance(neg, list) or len(neg) < 1:
                skipped += 1; continue

            clean_negs = [n.strip() for n in neg if isinstance(n, str) and n.strip()]
            if not clean_negs:
                skipped += 1; continue

            valid.append({
                "query":  query.strip(),
                "pos":    [pos[0].strip()],
                "neg":    clean_negs,
                "source": source_name,
            })

    log.info("Loaded %d valid triplets from %-45s (skipped %d)",
             len(valid), path.name, skipped)
    return valid


# ── Convert triplets to flat pairs ────────────────────────────────────────────
def triplets_to_pairs(triplets: list) -> list:
    pairs = []
    for item in triplets:
        query    = item["query"]
        pos_text = item["pos"][0]
        source   = item["source"]
        pos_cat  = get_law_category(pos_text)

        pairs.append({
            "query":        query,
            "document":     pos_text,
            "label":        1.0,
            "source":       source,
            "is_cross_code": False,
        })
        for neg_text in item["neg"]:
            neg_cat  = get_law_category(neg_text)
            is_cross = (pos_cat != neg_cat
                        and not pos_cat.startswith(neg_cat[:10])
                        and not neg_cat.startswith(pos_cat[:10]))
            pairs.append({
                "query":        query,
                "document":     neg_text,
                "label":        0.0,
                "source":       source,
                "is_cross_code": is_cross,
            })
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    log.info("=" * 60)
    log.info("  prepare_reranker_data.py  —  Script 1 / 4")
    log.info("=" * 60)

    # STEP 1 + 2 — Load, validate, and oversample at triplet level
    all_triplets = []
    source_counts = {}
    for source_name, path, weight in SOURCES:
        raw        = load_and_validate_jsonl(path, source_name)
        oversampled = raw * weight
        source_counts[f"{source_name}_{weight}x"] = len(oversampled)
        all_triplets.extend(oversampled)
        log.info("  -> %s: %d raw x %d = %d triplets",
                 source_name, len(raw), weight, len(oversampled))

    log.info("Total triplets after oversampling: %d", len(all_triplets))
    if not all_triplets:
        log.error("No triplets loaded — aborting.")
        sys.exit(1)

    # STEP 3 — Convert to flat pairs
    all_pairs = triplets_to_pairs(all_triplets)
    log.info("Total pairs after expansion: %d", len(all_pairs))

    # STEP 4 — Query-level train/val split
    query_to_pairs = defaultdict(list)
    for pair in all_pairs:
        query_to_pairs[pair["query"]].append(pair)

    unique_queries = sorted(query_to_pairs.keys())
    random.shuffle(unique_queries)

    n_val         = max(10, int(len(unique_queries) * VAL_RATIO))
    val_queries   = set(unique_queries[-n_val:])
    train_queries = set(unique_queries[:-n_val])

    # Assert no overlap
    overlap = train_queries & val_queries
    assert len(overlap) == 0, f"Query overlap detected between train/val: {len(overlap)} queries"

    train_pairs, val_pairs = [], []
    for q, pairs in query_to_pairs.items():
        if q in train_queries:
            train_pairs.extend(pairs)
        else:
            val_pairs.extend(pairs)

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    # Verify both sets have pos and neg
    train_labels = {p["label"] for p in train_pairs}
    val_labels   = {p["label"] for p in val_pairs}
    assert 1.0 in train_labels and 0.0 in train_labels, "Train set missing pos or neg pairs"
    assert 1.0 in val_labels   and 0.0 in val_labels,   "Val set missing pos or neg pairs"
    assert len(val_queries) >= 10, f"Val set has only {len(val_queries)} queries, need >= 10"

    pos_train    = sum(1 for p in train_pairs if p["label"] == 1.0)
    neg_train    = len(train_pairs) - pos_train
    pos_ratio    = pos_train / max(len(train_pairs), 1)
    cross_count  = sum(1 for p in all_pairs if p["is_cross_code"])

    log.info("=" * 60)
    log.info("  SPLIT SUMMARY")
    log.info("  Unique queries:   %d total -> %d train, %d val",
             len(unique_queries), len(train_queries), len(val_queries))
    log.info("  Train pairs:      %d  (pos=%d, neg=%d, ratio=%.3f)",
             len(train_pairs), pos_train, neg_train, pos_ratio)
    log.info("  Val pairs:        %d", len(val_pairs))
    log.info("  Cross-code pairs: %d", cross_count)
    log.info("=" * 60)

    if pos_ratio < 0.10 or pos_ratio > 0.35:
        log.warning("Positive ratio %.3f outside expected [0.10, 0.35] — check data", pos_ratio)

    # STEP 5 — Save outputs
    def save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        log.info("Saved -> %s  (%d items)", Path(path).name,
                 len(obj) if isinstance(obj, list) else 1)

    save_json(train_pairs, TRAIN_OUT)
    save_json(val_pairs,   VAL_OUT)

    stats = {
        "total_pairs":           len(all_pairs),
        "train_pairs":           len(train_pairs),
        "val_pairs":             len(val_pairs),
        "positive_pairs_train":  pos_train,
        "negative_pairs_train":  neg_train,
        "positive_ratio_train":  round(pos_ratio, 4),
        "unique_queries_train":  len(train_queries),
        "unique_queries_val":    len(val_queries),
        "cross_code_pairs":      cross_count,
        "source_breakdown":      source_counts,
    }
    save_json(stats, STATS_OUT)

    # STEP 6 — Build TripletEvaluator dataset from VAL pairs only
    log.info("Building TripletEvaluator dataset from val pairs...")
    val_q_to_pos  = {}
    val_q_to_negs = defaultdict(list)

    for p in val_pairs:
        q = p["query"]
        if p["label"] == 1.0:
            val_q_to_pos[q] = p["document"]
        else:
            val_q_to_negs[q].append(p["document"])

    triplet_eval_records = []
    for q, pos_doc in val_q_to_pos.items():
        negs = val_q_to_negs.get(q, [])
        if not negs:
            continue
        # Hardest negative: closest in length to positive (proxy for similarity)
        hardest_neg = min(negs, key=lambda n: abs(len(n) - len(pos_doc)))
        triplet_eval_records.append({
            "anchor":   q,
            "positive": pos_doc,
            "negative": hardest_neg,
        })

    if len(triplet_eval_records) < 10:
        log.warning("TripletEvaluator has only %d triplets — need >= 10",
                    len(triplet_eval_records))
    else:
        log.info("TripletEvaluator: %d triplets built from val set",
                 len(triplet_eval_records))

    save_json(triplet_eval_records, TRIPLET_EVAL)

    log.info("=" * 60)
    log.info("  DONE — Pre-training checks:")
    log.info("  positive_ratio_train = %.3f  (expected 0.10-0.35)", pos_ratio)
    log.info("  unique_queries_val   = %d    (expected >= 10)", len(val_queries))
    log.info("  triplet_eval size    = %d    (expected >= 10)", len(triplet_eval_records))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
