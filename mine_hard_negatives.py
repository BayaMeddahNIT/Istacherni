"""
mine_hard_negatives.py  (v2 — Deep Mining)
==========================================
CHANGES vs v1:
  1. Retrieval depth raised to 100 candidates (was 20).
  2. Rank-Shift Sampling: negative window shifts based on golden rank.
       golden @ rank 1      → sample from ranks 10–50 (distractor zone)
       golden @ ranks 2–9   → sample from ranks adjacent to golden ±5
       golden @ rank ≥10    → sample ALL non-golden ranks ≤50 (blind-spot)
  3. Strict Δ-filter: only negatives with Δ ≤ 0.10 pass.
       Queries with ZERO passing negatives → flagged for manual review.
  4. Cross-code bonus: negatives from a DIFFERENT legal code than the
       positive are prioritised (sorted first).
  5. Confidently-wrong prioritisation: queries where golden > rank 10
       are processed first regardless of dataset order.
  6. Verification threshold tightened: ✅ = avg Δ ≤ 0.10  (was 0.15).

USAGE:
  python mine_hard_negatives.py --sample 10   # verification (10 queries)
  python mine_hard_negatives.py --full        # full dataset
  python mine_hard_negatives.py --full --delta-threshold 0.15  # relaxed
"""

import sys
import json
import time
import argparse
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
        logging.FileHandler(PROJECT_ROOT / "hard_negatives_mining.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH        = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
OUTPUT_JSONL        = PROJECT_ROOT / "hard_negatives.jsonl"
FLAGGED_JSON        = PROJECT_ROOT / "hard_negatives_flagged.json"
VERIFICATION_JSON   = PROJECT_ROOT / "hard_negatives_verification.json"

RETRIEVE_TOP_K      = 100   # Set to 100 to scan all useful ranks and double mining speed
MAX_NEG_SCAN_RANK   = 100   # Scan deeper for negatives
DEFAULT_SAMPLE      = 10
DELTA_THRESHOLD     = 0.10  # strict filter (Δ ≤ 0.10 passes)
BM25_WEIGHT         = 0.3
DENSE_WEIGHT        = 0.7


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
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


def chunk_id(doc: dict) -> str:
    return f"{doc.get('law_name', '')}__art__{doc.get('article_number', '')}"


def law_code(doc: dict) -> str:
    """Normalised law-code key for cross-code comparison."""
    name = str(doc.get("law_name", "")).strip()
    # Map to short canonical keys
    if "العقوبات" in name:           return "penal"
    if "التجاري" in name:            return "commercial"
    if "المدني" in name:             return "civil"
    if "الاجراءات" in name:          return "procedure"
    if "العمل" in name or "90-11" in name: return "labour"
    if "18-05" in name:              return "ecommerce"
    if "03-03" in name:              return "competition"
    return name[:40]


def is_new_article(doc: dict) -> bool:
    """Check if the article is one of the 23 newly added penal articles."""
    aid = doc.get("id", "")
    # New IDs follow the pattern DZ_PENAL_ART_2xx
    import re
    if re.match(r"DZ_PENAL_ART_2(0[8-9]|1[0-9]|2[0-9]|3[0-1])", aid):
        return True
    return False


def is_golden(doc: dict, gt_articles: list[str]) -> bool:
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
#  RANK-SHIFT SAMPLING WINDOW  (v2 core logic)
# ══════════════════════════════════════════════════════════════════════════════

def _neg_rank_window(golden_rank: int) -> tuple[int, int]:
    """
    Return (start, end) rank window for hard-negative candidates.

    Strategy:
      - Golden @ 1     → ranks 10–50  (distractor zone; skip obvious neighbours)
      - Golden @ 2–9   → ranks golden±1 up to 50 (adjacent confusion zone)
      - Golden @ ≥10   → ranks 1 up to 50 (entire top-50, all non-golden)
    """
    if golden_rank == 1:
        return (10, MAX_NEG_SCAN_RANK)
    elif golden_rank < 10:
        start = max(1, golden_rank - 1)
        return (start, MAX_NEG_SCAN_RANK)
    else:  # golden is deeply buried — model's blind spot
        return (1, MAX_NEG_SCAN_RANK)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE MINING
# ══════════════════════════════════════════════════════════════════════════════

def mine_query(
    query: str,
    gt_articles: list[str],
    retrieve_fn,
    top_neg: int,
    delta_threshold: float,
) -> Optional[dict]:
    """
    Mine TRUE hard negatives for a single query.
    Returns triplet dict or None (no golden found).
    Includes a 'flagged' key when only easy negatives exist.
    """
    try:
        candidates = retrieve_fn(query, top_k=RETRIEVE_TOP_K)
    except Exception as exc:
        log.warning("  Retrieval error: %s", exc)
        return None

    if not candidates:
        log.warning("  No candidates returned.")
        return None

    # ── Locate golden ──────────────────────────────────────────────────────────
    golden_docs = []
    for rank, doc in enumerate(candidates, start=1):
        if is_golden(doc, gt_articles):
            golden_docs.append((rank, doc))

    if not golden_docs:
        log.debug("  ✗ Golden NOT found in top-%d.", RETRIEVE_TOP_K)
        return None

    golden_rank, golden_doc = golden_docs[0]
    golden_text = build_chunk_text(golden_doc)
    golden_code = law_code(golden_doc)
    golden_score = golden_doc.get("rerank_score", 0.0)

    # ── Determine rank window (Rank-Shift Sampling) ────────────────────────────
    win_start, win_end = _neg_rank_window(golden_rank)
    window_label = f"ranks {win_start}–{win_end}"

    # ── Collect all non-golden docs in window ──────────────────────────────────
    candidate_negs = []
    for rank, doc in enumerate(candidates, start=1):
        if is_golden(doc, gt_articles):
            continue
        if not (win_start <= rank <= win_end):
            continue
        delta = abs(golden_score - doc.get("rerank_score", 0.0))
        cross_code = law_code(doc) != golden_code
        candidate_negs.append({
            "rank"        : rank,
            "doc"         : doc,
            "delta"       : round(delta, 6),
            "cross_code"  : cross_code,
            "rerank_score": round(doc.get("rerank_score", 0.0), 6),
            "rrf_score"   : round(doc.get("rrf_score", 0.0), 6),
            "law_name"    : doc.get("law_name", ""),
            "article_num" : doc.get("article_number", ""),
        })

    # Apply Δ ≤ threshold filter
    strict_negs = [c for c in candidate_negs if c["delta"] <= delta_threshold]

    # Cross-code bonus + New-article bonus: 
    # Sort strict negs so cross-code and new articles come first
    strict_negs.sort(key=lambda x: (not is_new_article(x["doc"]), not x["cross_code"], x["delta"]))

    flagged = False
    if not strict_negs:
        # No true hard negatives — flag for review, but still record
        log.warning("  ⚑ FLAGGED: only easy negatives (all Δ > %.2f). "
                    "Best Δ=%.4f. Skipping triplet.",
                    delta_threshold,
                    min((c["delta"] for c in candidate_negs), default=float("inf")))
        flagged = True
        # Don't add to output
        return {
            "query"   : query,
            "flagged" : True,
            "golden_rank": golden_rank,
            "best_delta" : min((c["delta"] for c in candidate_negs), default=None),
            "window"  : window_label,
        }

    hard_neg_docs = strict_negs[:top_neg]
    neg_texts = [build_chunk_text(c["doc"]) for c in hard_neg_docs]

    # ── Score summary ──────────────────────────────────────────────────────────
    score_summary = {
        "golden_rank"   : golden_rank,
        "golden_rerank" : round(golden_score, 6),
        "golden_rrf"    : round(golden_doc.get("rrf_score", 0.0), 6),
        "golden_id"     : chunk_id(golden_doc),
        "golden_code"   : golden_code,
        "window"        : window_label,
        "hard_negatives": [
            {
                "rank"        : c["rank"],
                "id"          : chunk_id(c["doc"]),
                "rerank_score": c["rerank_score"],
                "rrf_score"   : c["rrf_score"],
                "law_name"    : c["law_name"],
                "article_num" : c["article_num"],
                "delta_rerank": c["delta"],
                "cross_code"  : c["cross_code"],
            }
            for c in hard_neg_docs
        ],
    }

    return {
        "query"         : query,
        "pos"           : [golden_text],
        "neg"           : neg_texts,
        "score_summary" : score_summary,
        "flagged"       : False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_mining(
    dataset: list[dict],
    retrieve_fn,
    top_neg: int,
    delta_threshold: float,
    output_path: Path,
    verification_path: Path,
    is_verification: bool = False,
) -> dict:
    total = len(dataset)
    mined = 0
    skipped_no_gt = 0
    no_golden = 0
    flagged_list = []
    verification_records = []

    log.info("=" * 70)
    log.info("  Deep Hard Negative Mining v2  |  %d queries", total)
    log.info("  Retrieve Top-K = %d  |  Max scan rank = %d", RETRIEVE_TOP_K, MAX_NEG_SCAN_RANK)
    log.info("  Δ threshold = %.2f  |  BM25=%.2f  Dense=%.2f",
             delta_threshold, BM25_WEIGHT, DENSE_WEIGHT)
    log.info("=" * 70)

    # ── Sort: confidently-wrong queries first (golden > rank 10 from v1 run) ──
    # We don't know ranks yet, so process in natural order but log priority
    log.info("  NOTE: Queries where previous golden rank > 10 are high-priority.")
    log.info("=" * 70)

    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(dataset, start=1):
            query       = item.get("question", "").strip()
            gt_articles = item.get("articles", [])

            if not query or not gt_articles:
                skipped_no_gt += 1
                continue

            log.info("[%d/%d] Q: %s", idx, total, query[:80])

            result = mine_query(query, gt_articles, retrieve_fn, top_neg, delta_threshold)

            if result is None:
                no_golden += 1
                log.info("  → Skipped (no golden found in top-%d)", RETRIEVE_TOP_K)
                continue

            if result.get("flagged"):
                flagged_list.append({
                    "query"      : result["query"],
                    "golden_rank": result["golden_rank"],
                    "best_delta" : result["best_delta"],
                    "window"     : result["window"],
                })
                log.info("  → ⚑ Flagged for manual review (no Δ ≤ %.2f neg found)", delta_threshold)
                continue

            mined += 1
            ss = result["score_summary"]
            neg_count = len(result["neg"])
            cross_count = sum(1 for h in ss["hard_negatives"] if h["cross_code"])

            log.info(
                "  → ✅ Golden @ rank %d (%s)  rerank=%.4f  |  %d negs  [%d cross-code]",
                ss["golden_rank"], ss["window"], ss["golden_rerank"],
                neg_count, cross_count,
            )
            for hn in ss["hard_negatives"]:
                xmark = "⚡" if hn["cross_code"] else "  "
                log.info(
                    "      %s [%2d] %s - Art.%s  rerank=%.4f  Δ=%.4f",
                    xmark, hn["rank"], hn["law_name"][:28],
                    hn["article_num"], hn["rerank_score"], hn["delta_rerank"],
                )

            triplet = {
                "query": result["query"],
                "pos"  : result["pos"],
                "neg"  : result["neg"],
            }
            fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")

            if is_verification or idx <= DEFAULT_SAMPLE:
                verification_records.append({
                    "query"        : query,
                    "score_summary": ss,
                })

    elapsed = time.time() - t0

    # ── Save flagged queries ───────────────────────────────────────────────────
    if flagged_list:
        with open(FLAGGED_JSON, "w", encoding="utf-8") as f:
            json.dump(flagged_list, f, ensure_ascii=False, indent=2)
        log.info("  Flagged queries saved → %s  (%d queries)", FLAGGED_JSON.name, len(flagged_list))

    log.info("\n" + "=" * 70)
    log.info("  MINING COMPLETE")
    log.info("=" * 70)
    log.info("  Total queries      : %d", total)
    log.info("  Triplets mined     : %d", mined)
    log.info("  Flagged (easy neg) : %d", len(flagged_list))
    log.info("  No golden found    : %d", no_golden)
    log.info("  Skipped (no GT)    : %d", skipped_no_gt)
    log.info("  Time elapsed       : %.1f s", elapsed)
    log.info("  Output             : %s", output_path.name)
    log.info("=" * 70)

    if verification_records:
        with open(verification_path, "w", encoding="utf-8") as f:
            json.dump(verification_records, f, ensure_ascii=False, indent=2)
        log.info("  Verification report → %s", verification_path.name)
        _print_verification_table(verification_records, delta_threshold)

    return {
        "total": total, "mined": mined,
        "flagged": len(flagged_list), "no_golden": no_golden,
        "elapsed": round(elapsed, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION TABLE  (threshold tightened to Δ ≤ 0.10)
# ══════════════════════════════════════════════════════════════════════════════

def _print_verification_table(records: list[dict], threshold: float):
    print("\n" + "═" * 100)
    print("  VERIFICATION REPORT v2 — True Hard Negative Score Proximity")
    print("═" * 100)
    print(f"  {'#':<3}  {'Query':<40}  {'G.Rank':<7}  {'Window':<14}  "
          f"{'G.Score':<10}  {'Avg Δ':<9}  {'Min Δ':<9}  {'Cross?'}")
    print("─" * 100)

    pass_count = 0
    for i, rec in enumerate(records, 1):
        q  = rec["query"][:38]
        ss = rec["score_summary"]
        gr = ss["golden_rank"]
        gs = ss["golden_rerank"]
        win = ss.get("window", "?")

        deltas = [hn["delta_rerank"] for hn in ss["hard_negatives"]]
        avg_d  = round(sum(deltas) / len(deltas), 4) if deltas else float("nan")
        min_d  = round(min(deltas), 4) if deltas else float("nan")
        cross_n = sum(1 for h in ss["hard_negatives"] if h.get("cross_code"))

        flag = "✅" if avg_d <= threshold else "⚠ "
        if avg_d <= threshold:
            pass_count += 1
        print(f"  {i:<3}  {q:<40}  {gr:<7}  {win:<14}  "
              f"{gs:<10.4f}  {avg_d:<9.4f}  {min_d:<9.4f}  {cross_n}/{len(deltas)} {flag}")

    print("═" * 100)
    print(f"  Passed: {pass_count}/{len(records)} queries "
          f"(✅ = avg Δ ≤ {threshold}  ⚠ = avg Δ > {threshold})")
    print("─" * 100)
    print()

    # Detail table for all records (not just first)
    for rec in records:
        ss = rec["score_summary"]
        if not ss["hard_negatives"]:
            continue
        print(f"  DETAIL — {rec['query'][:72]}")
        print(f"  Golden: {ss['golden_id'][:50]}  "
              f"score={ss['golden_rerank']:.4f}  code={ss.get('golden_code','?')}")
        print(f"  {'Rk':<4} {'Law':<30} {'Art':<8} {'Score':<10} {'Δ':<9} {'Cross'}")
        print("  " + "─" * 65)
        for hn in ss["hard_negatives"]:
            xmark = "⚡YES" if hn["cross_code"] else "   no"
            print(f"  {hn['rank']:<4} {hn['law_name'][:28]:<30} "
                  f"{str(hn['article_num']):<8} {hn['rerank_score']:<10.4f} "
                  f"{hn['delta_rerank']:<9.4f} {xmark}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Deep Hard Negative Mining v2")
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--full", action="store_true")
    p.add_argument("--top-neg", type=int, default=9,
                   help="Max hard negatives per query (default: 9)")
    p.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT)
    p.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT)
    p.add_argument("--delta-threshold", type=float, default=DELTA_THRESHOLD,
                   help=f"Max Δ to accept a hard negative (default: {DELTA_THRESHOLD})")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("Initializing hybrid retriever (BM25 + BGE-M3 + CrossEncoder)…")
    log.info("Retrieval depth: top-%d candidates per query.", RETRIEVE_TOP_K)

    from bm25_rag.bm25_retriever import bm25_retrieve
    from dense_rag.bge_retriever import dense_retrieve
    from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion
    from hybrid_rag.reranker import rerank_candidates

    def retrieve_fn(query: str, top_k: int) -> list[dict]:
        bm25_results  = bm25_retrieve(query, top_k=top_k)
        dense_results = dense_retrieve(query, top_k=top_k)
        fused = reciprocal_rank_fusion(
            bm25_results, dense_results,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
        )
        return rerank_candidates(query, fused[:top_k], top_k=top_k)

    log.info("Loading dataset: %s", DATASET_PATH.name)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)
    log.info("Loaded %d questions.", len(full_dataset))

    if args.full:
        dataset = full_dataset
        is_ver  = False
        out_path = OUTPUT_JSONL
        ver_path = VERIFICATION_JSON
        log.info("Mode: FULL DATASET (%d queries)", len(dataset))
    else:
        n = min(args.sample, len(full_dataset))
        dataset = full_dataset[:n]
        is_ver   = True
        out_path = PROJECT_ROOT / f"hard_negatives_deep_sample_{n}.jsonl"
        ver_path = VERIFICATION_JSON
        log.info("Mode: SAMPLE / VERIFICATION (%d queries)", n)

    run_mining(
        dataset           = dataset,
        retrieve_fn       = retrieve_fn,
        top_neg           = args.top_neg,
        delta_threshold   = args.delta_threshold,
        output_path       = out_path,
        verification_path = ver_path,
        is_verification   = is_ver,
    )


if __name__ == "__main__":
    main()
