"""
step2_failure_mining.py
=======================
Mines failure-specific hard negatives for the queries that the model
currently gets wrong (precision_at_1 == 0).

Input:
    retrieval_evaluation_results.csv
    The existing retrieval pipeline

For each failing query:
    1. Re-run hybrid retrieval + reranking (top-30)
    2. Identify wrong rank-1 article  →  hard negative
    3. Identify gold article          →  positive
    4. Also add rank #2 and #3 as extra negatives if Δ from gold ≤ 0.20

Output:
    hard_negatives_failures.jsonl
    failure_analysis.json

USAGE:
    python step2_failure_mining.py
"""

import sys
import csv
import json
import time
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "step2_failure_mining.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EVAL_CSV        = PROJECT_ROOT / "retrieval_evaluation_results.csv"
DATASET_PATH    = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
OUTPUT_JSONL    = PROJECT_ROOT / "hard_negatives_failures.jsonl"
ANALYSIS_JSON   = PROJECT_ROOT / "failure_analysis.json"

RETRIEVE_TOP_K  = 30
EXTRA_NEG_DELTA = 0.20   # relaxed delta for ranks #2 and #3
BM25_WEIGHT     = 0.3
DENSE_WEIGHT    = 0.7


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — copied verbatim from mine_hard_negatives.py
# ══════════════════════════════════════════════════════════════════════════════

def _safe(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v is not None)
    return str(val)


def build_chunk_text(doc: dict) -> str:
    law_name    = _safe(doc.get("law_name"))
    art_num     = _safe(doc.get("article_number"))
    title       = _safe(doc.get("title"))
    keywords    = _safe(doc.get("keywords"))
    original    = _safe(doc.get("text_original"))
    explanation = _safe(doc.get("text_explanation"))
    summary     = _safe(doc.get("summary"))
    header = f"[{law_name} - المادة {art_num}]" if law_name and art_num else ""
    parts  = [p for p in [header, title, keywords, original, explanation, summary] if p]
    return " | ".join(parts).strip() or "unknown"


def article_label(doc: dict) -> str:
    law  = doc.get("law_name", "")
    num  = doc.get("article_number", "")
    return f"{law} - المادة {num}"


def is_golden(doc: dict, gt_articles: list) -> bool:
    c_law = str(doc.get("law_name", ""))
    c_num = str(doc.get("article_number", "")).strip()
    for gt in gt_articles:
        parts = gt.split(" - المادة ")
        if len(parts) == 2:
            gt_law = parts[0].strip()
            gt_num = parts[1].strip()
        else:
            gt_law = ""
            gt_num = "".join(filter(str.isdigit, gt))
        if c_num == gt_num:
            if (not gt_law or gt_law in c_law or c_law in gt_law
                    or ("قانون" in gt_law and "قانون" in c_law)):
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD CSV AND IDENTIFY FAILURES
# ══════════════════════════════════════════════════════════════════════════════

def _unquote(val: str) -> str:
    """Strip outer single-quotes that the CSV writer may have added."""
    v = val.strip()
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        v = v[1:-1]
    return v.strip()


def load_failing_queries() -> list[dict]:
    """
    Load CSV and return rows where precision_at_1 == 0.

    Handles two quirks in retrieval_evaluation_results.csv:
      1. UTF-8 BOM on first column → use encoding='utf-8-sig'
      2. Cell values wrapped in single-quotes → stripped by _unquote()
    """
    if not EVAL_CSV.exists():
        log.error("Evaluation CSV not found: %s", EVAL_CSV)
        sys.exit(1)

    failing = []
    with open(EVAL_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip single-quote wrappers from every cell
            row = {k: _unquote(v) for k, v in row.items()}
            try:
                p1 = float(row.get("precision_at_1", "1") or "1")
            except (ValueError, TypeError):
                continue
            if p1 == 0:
                gt_raw = row.get("expected_articles", "")
                gt_articles = [a.strip() for a in gt_raw.split("|") if a.strip()]
                query = row.get("question", "").strip()
                if not query:
                    log.warning("  Skipping row with empty question (gt=%s)", gt_raw[:50])
                    continue
                failing.append({
                    "query"      : query,
                    "gt_articles": gt_articles,
                    "mrr"        : float(row.get("reciprocal_rank", "0") or "0"),
                    "hit_rate"   : int(row.get("is_hit", "false").lower() == "true"),
                })

    log.info("Found %d failing queries (precision_at_1 = 0).", len(failing))
    return failing


# ══════════════════════════════════════════════════════════════════════════════
#  MINE FAILURE-SPECIFIC NEGATIVES
# ══════════════════════════════════════════════════════════════════════════════

def mine_failure(
    query: str,
    gt_articles: list,
    retrieve_fn,
) -> Optional[dict]:
    """
    For a failing query:
    1. Re-run hybrid retrieval + reranking (top-30)
    2. Find gold article and its rank
    3. Build triplet with rank-1 wrong answer as primary negative
    4. Add ranks #2 and #3 as extra negatives if Δ ≤ 0.20

    Returns None if gold not found in top-30.
    """
    try:
        candidates = retrieve_fn(query, top_k=RETRIEVE_TOP_K)
    except Exception as exc:
        log.warning("  Retrieval error: %s", exc)
        return None

    if not candidates:
        log.warning("  No candidates returned.")
        return None

    # ── Identify rank-1 (the wrong answer) ────────────────────────────────────
    wrong_rank1     = candidates[0]
    wrong_rank1_txt = build_chunk_text(wrong_rank1)
    wrong_score     = wrong_rank1.get("rerank_score", 0.0)

    # ── Find gold article in candidates ───────────────────────────────────────
    gold_doc   = None
    gold_rank  = None
    gold_score = None

    for rank, doc in enumerate(candidates, start=1):
        if is_golden(doc, gt_articles):
            gold_doc   = doc
            gold_rank  = rank
            gold_score = doc.get("rerank_score", 0.0)
            break

    if gold_doc is None:
        log.warning("  ✗ Gold NOT found in top-%d for this query.", RETRIEVE_TOP_K)
        # Use GT article text from dataset as positive (fallback)
        # We still build the triplet with wrong rank-1 as negative
        # but flag it as gold_not_in_top30
        gold_text  = " | ".join(gt_articles[:1])  # minimal fallback
        gold_rank  = -1
        gold_score = 0.0
        gold_doc   = {}
    else:
        gold_text = build_chunk_text(gold_doc)

    # ── Primary negative: wrong rank-1 ────────────────────────────────────────
    negatives     = [wrong_rank1_txt]
    neg_meta      = [{"rank": 1, "label": article_label(wrong_rank1),
                      "score": round(wrong_score, 6), "delta": None}]

    # ── Extra negatives: ranks #2 and #3 ──────────────────────────────────────
    for extra_rank_idx in range(1, 3):  # indices 1 and 2 → ranks 2 and 3
        if extra_rank_idx >= len(candidates):
            break
        extra_doc  = candidates[extra_rank_idx]
        extra_score = extra_doc.get("rerank_score", 0.0)

        if gold_score is not None and gold_score > 0:
            delta = abs(gold_score - extra_score)
        else:
            delta = float("inf")

        if is_golden(extra_doc, gt_articles):
            continue  # skip if it is also a gold article

        if delta <= EXTRA_NEG_DELTA:
            negatives.append(build_chunk_text(extra_doc))
            neg_meta.append({
                "rank" : extra_rank_idx + 1,
                "label": article_label(extra_doc),
                "score": round(extra_score, 6),
                "delta": round(delta, 6),
            })
            log.info(
                "  + Extra neg @ rank %d  Δ=%.4f  %s",
                extra_rank_idx + 1, delta, article_label(extra_doc)[:40],
            )
        else:
            log.info(
                "  - Skip extra neg @ rank %d  Δ=%.4f > %.2f",
                extra_rank_idx + 1, delta, EXTRA_NEG_DELTA,
            )

    # ── Compute delta for primary neg ─────────────────────────────────────────
    primary_delta = abs(gold_score - wrong_score) if gold_score else None
    neg_meta[0]["delta"] = round(primary_delta, 6) if primary_delta is not None else None

    analysis = {
        "query"           : query,
        "golden_article"  : article_label(gold_doc) if gold_doc else gt_articles[0] if gt_articles else "",
        "golden_rank"     : gold_rank,
        "golden_score"    : round(gold_score, 6) if gold_score else 0.0,
        "wrong_top1"      : article_label(wrong_rank1),
        "wrong_top1_score": round(wrong_score, 6),
        "delta"           : round(primary_delta, 6) if primary_delta is not None else None,
        "negatives_added" : len(negatives),
        "neg_meta"        : neg_meta,
    }

    triplet = {
        "query": query,
        "pos"  : [gold_text],
        "neg"  : negatives,
    }

    return {"triplet": triplet, "analysis": analysis}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("  Step 2 — Failure-Specific Hard Negative Mining")
    log.info("  Extra neg Δ threshold = %.2f  |  top-K = %d", EXTRA_NEG_DELTA, RETRIEVE_TOP_K)
    log.info("=" * 70)

    # ── Load failing queries from CSV ─────────────────────────────────────────
    failing_queries = load_failing_queries()
    if not failing_queries:
        log.info("No failing queries found — nothing to mine.")
        return

    # ── Init retrieval pipeline ───────────────────────────────────────────────
    log.info("Initialising retrieval pipeline…")
    from bm25_rag.bm25_retriever import bm25_retrieve
    from dense_rag.bge_retriever import dense_retrieve
    from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion
    from hybrid_rag.reranker import rerank_candidates

    def retrieve_fn(query: str, top_k: int) -> list:
        bm25_results  = bm25_retrieve(query, top_k=top_k)
        dense_results = dense_retrieve(query, top_k=top_k)
        fused = reciprocal_rank_fusion(
            bm25_results, dense_results,
            bm25_weight=BM25_WEIGHT,
            dense_weight=DENSE_WEIGHT,
        )
        return rerank_candidates(query, fused[:top_k], top_k=top_k)

    # ── Mine each failure ─────────────────────────────────────────────────────
    all_analyses = []
    total = len(failing_queries)
    mined = 0

    t0 = time.time()

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(failing_queries, start=1):
            query       = item["query"]
            gt_articles = item["gt_articles"]

            if not query:
                continue

            log.info("[%d/%d] Q: %s", idx, total, query[:80])
            log.info("  GT articles: %s", " | ".join(gt_articles[:3]))

            result = mine_failure(query, gt_articles, retrieve_fn)

            if result is None:
                log.warning("  → Skipped (retrieval error).")
                continue

            analysis = result["analysis"]
            triplet  = result["triplet"]
            all_analyses.append(analysis)

            log.info(
                "  → ✅ Gold @ rank %d (score=%.4f)  |  Wrong#1: %s (score=%.4f)  "
                "Δ=%.4f  |  %d negatives",
                analysis["golden_rank"],
                analysis["golden_score"],
                analysis["wrong_top1"][:35],
                analysis["wrong_top1_score"],
                analysis["delta"] or 0.0,
                analysis["negatives_added"],
            )

            fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            mined += 1

    elapsed = time.time() - t0

    # ── Save analysis JSON ─────────────────────────────────────────────────────
    with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  STEP 2 COMPLETE")
    log.info("=" * 70)
    log.info("  Failing queries processed  : %d", total)
    log.info("  Triplets mined             : %d", mined)
    log.info("  Elapsed                    : %.1f s", elapsed)
    log.info("  Triplets → %s", OUTPUT_JSONL.name)
    log.info("  Analysis → %s", ANALYSIS_JSON.name)
    log.info("=" * 70)

    # ── Failure analysis table ─────────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("  STEP 2 — FAILURE ANALYSIS TABLE")
    print("═" * 110)
    print(f"  {'#':<3}  {'Query':<38}  {'GoldRank':<9}  {'GoldScore':<10}  "
          f"{'WrongTop1':<35}  {'Delta':<8}  {'Negs'}")
    print("─" * 110)
    for i, rec in enumerate(all_analyses, 1):
        q_short = rec["query"][:36]
        w1_short = rec["wrong_top1"][:33]
        print(
            f"  {i:<3}  {q_short:<38}  {rec['golden_rank']:<9}  "
            f"{rec['golden_score']:<10.4f}  {w1_short:<35}  "
            f"{(rec['delta'] or 0.0):<8.4f}  {rec['negatives_added']}"
        )
    print("═" * 110)
    print(f"  Total triplets: {mined}  |  Total negatives written: "
          f"{sum(r['negatives_added'] for r in all_analyses)}")
    print()


if __name__ == "__main__":
    main()
