"""
step1_relaxed_mining.py
=======================
Re-runs hard negative mining ONLY for the 74 queries that were flagged
(no negative found at Δ ≤ 0.10) using a relaxed Δ threshold of 0.15.

Everything else is identical to mine_hard_negatives.py:
  - Retrieve top-200 candidates
  - Same rank-shift sampling windows
  - Cross-code bonus (cross-code negatives sorted first)
  - top_neg = 9 max per query

Outputs:
  - hard_negatives_relaxed.jsonl        (recovered queries)
  - hard_negatives_still_flagged.json   (still no negative at Δ ≤ 0.15)

USAGE:
    python step1_relaxed_mining.py
"""

import sys
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
        logging.FileHandler(PROJECT_ROOT / "step1_relaxed_mining.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
FLAGGED_JSON        = PROJECT_ROOT / "hard_negatives_flagged.json"
DATASET_PATH        = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
OUTPUT_JSONL        = PROJECT_ROOT / "hard_negatives_relaxed.jsonl"
STILL_FLAGGED_JSON  = PROJECT_ROOT / "hard_negatives_still_flagged.json"

RETRIEVE_TOP_K      = 200
MAX_NEG_SCAN_RANK   = 100
DELTA_THRESHOLD     = 0.15   # relaxed from 0.10
TOP_NEG             = 9
BM25_WEIGHT         = 0.3
DENSE_WEIGHT        = 0.7


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


def chunk_id(doc: dict) -> str:
    return f"{doc.get('law_name', '')}__art__{doc.get('article_number', '')}"


def law_code(doc: dict) -> str:
    """Normalised law-code key for cross-code comparison."""
    name = str(doc.get("law_name", "")).strip()
    if "العقوبات" in name:                    return "penal"
    if "التجاري" in name:                     return "commercial"
    if "المدني" in name:                      return "civil"
    if "الاجراءات" in name:                   return "procedure"
    if "العمل" in name or "90-11" in name:    return "labour"
    if "18-05" in name:                       return "ecommerce"
    if "03-03" in name:                       return "competition"
    return name[:40]


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


def _neg_rank_window(golden_rank: int) -> tuple:
    if golden_rank == 1:
        return (10, MAX_NEG_SCAN_RANK)
    elif golden_rank < 10:
        start = max(1, golden_rank - 1)
        return (start, MAX_NEG_SCAN_RANK)
    else:
        return (1, MAX_NEG_SCAN_RANK)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE MINING  (relaxed threshold variant)
# ══════════════════════════════════════════════════════════════════════════════

def mine_query_relaxed(
    query: str,
    gt_articles: list,
    retrieve_fn,
    top_neg: int = TOP_NEG,
    delta_threshold: float = DELTA_THRESHOLD,
) -> Optional[dict]:
    """Mine hard negatives for a single query using a relaxed Δ threshold."""
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
        return {"status": "no_golden", "query": query}

    golden_rank, golden_doc = golden_docs[0]
    golden_text  = build_chunk_text(golden_doc)
    golden_code  = law_code(golden_doc)
    golden_score = golden_doc.get("rerank_score", 0.0)

    # ── Rank-shift sampling window ─────────────────────────────────────────────
    win_start, win_end = _neg_rank_window(golden_rank)
    window_label = f"ranks {win_start}–{win_end}"

    # ── Collect candidate negatives in window ──────────────────────────────────
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

    # Apply Δ ≤ relaxed threshold filter
    strict_negs = [c for c in candidate_negs if c["delta"] <= delta_threshold]

    # Cross-code bonus: sort cross-code negatives first, then by delta
    strict_negs.sort(key=lambda x: (not x["cross_code"], x["delta"]))

    best_delta = min((c["delta"] for c in candidate_negs), default=None)

    if not strict_negs:
        return {
            "status"      : "still_flagged",
            "query"       : query,
            "golden_rank" : golden_rank,
            "golden_score": round(golden_score, 6),
            "best_delta"  : best_delta,
            "window"      : window_label,
            "threshold_used": delta_threshold,
        }

    hard_neg_docs = strict_negs[:top_neg]
    neg_texts     = [build_chunk_text(c["doc"]) for c in hard_neg_docs]

    score_summary = {
        "golden_rank"   : golden_rank,
        "golden_rerank" : round(golden_score, 6),
        "golden_id"     : chunk_id(golden_doc),
        "golden_code"   : golden_code,
        "window"        : window_label,
        "threshold_used": delta_threshold,
        "hard_negatives": [
            {
                "rank"        : c["rank"],
                "id"          : chunk_id(c["doc"]),
                "rerank_score": c["rerank_score"],
                "law_name"    : c["law_name"],
                "article_num" : c["article_num"],
                "delta_rerank": c["delta"],
                "cross_code"  : c["cross_code"],
            }
            for c in hard_neg_docs
        ],
    }

    return {
        "status"       : "recovered",
        "query"        : query,
        "pos"          : [golden_text],
        "neg"          : neg_texts,
        "score_summary": score_summary,
        "golden_score" : round(golden_score, 6),
        "golden_rank"  : golden_rank,
        "best_delta"   : min(c["delta"] for c in hard_neg_docs),
        "negs_found"   : len(hard_neg_docs),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("  Step 1 — Relaxed Hard Negative Mining  (Δ ≤ %.2f)", DELTA_THRESHOLD)
    log.info("=" * 70)

    # ── Load flagged queries ───────────────────────────────────────────────────
    if not FLAGGED_JSON.exists():
        log.error("Flagged file not found: %s", FLAGGED_JSON)
        return
    with open(FLAGGED_JSON, "r", encoding="utf-8") as f:
        flagged_entries = json.load(f)
    log.info("Loaded %d flagged queries from %s", len(flagged_entries), FLAGGED_JSON.name)

    flagged_queries = {entry["query"] for entry in flagged_entries}

    # ── Load dataset to get ground-truth articles ──────────────────────────────
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)
    log.info("Loaded %d dataset entries for GT lookup.", len(full_dataset))

    # Build query → GT articles lookup
    query_to_gt: dict = {}
    for item in full_dataset:
        q = item.get("question", "").strip()
        arts = item.get("articles", [])
        if q and arts:
            query_to_gt[q] = arts

    # ── Build targeted dataset of only flagged queries ─────────────────────────
    targeted = []
    for fq in flagged_entries:
        q = fq["query"]
        gt = query_to_gt.get(q)
        if gt:
            targeted.append({"question": q, "articles": gt,
                             "prev_golden_rank": fq.get("golden_rank"),
                             "prev_best_delta" : fq.get("best_delta")})
        else:
            log.warning("  No GT found for flagged query: %s", q[:60])

    log.info("Targeting %d flagged queries with GT articles.", len(targeted))

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

    # ── Run relaxed mining ─────────────────────────────────────────────────────
    recovered       = []
    still_flagged   = []
    no_golden       = []

    t0 = time.time()
    total = len(targeted)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(targeted, start=1):
            query = item["question"]
            gt    = item["articles"]

            log.info("[%d/%d] Q: %s", idx, total, query[:80])
            log.info("  (prev rank=%s, prev best_Δ=%.4f)",
                     item.get("prev_golden_rank", "?"),
                     item.get("prev_best_delta") or 0.0)

            result = mine_query_relaxed(query, gt, retrieve_fn)

            if result is None or result.get("status") == "no_golden":
                no_golden.append(query)
                log.info("  → ✗ Golden NOT found in top-%d", RETRIEVE_TOP_K)
                continue

            status = result["status"]

            if status == "still_flagged":
                still_flagged.append({
                    "query"          : query,
                    "golden_rank"    : result["golden_rank"],
                    "golden_score"   : result["golden_score"],
                    "best_delta"     : result["best_delta"],
                    "threshold_tried": DELTA_THRESHOLD,
                    "window"         : result["window"],
                })
                log.info("  → ⚑ STILL FLAGGED  (best Δ=%.4f > %.2f)",
                         result["best_delta"] or 0.0, DELTA_THRESHOLD)
                continue

            # Recovered
            ss = result["score_summary"]
            neg_count   = result["negs_found"]
            cross_count = sum(1 for hn in ss["hard_negatives"] if hn["cross_code"])

            log.info(
                "  → ✅ RECOVERED  golden@rank %d  rerank=%.4f  |  %d negs  [%d cross-code]",
                result["golden_rank"], result["golden_score"],
                neg_count, cross_count,
            )
            for hn in ss["hard_negatives"]:
                xmark = "⚡" if hn["cross_code"] else "  "
                log.info(
                    "      %s [%2d] %s - Art.%s  rerank=%.4f  Δ=%.4f",
                    xmark, hn["rank"], hn["law_name"][:28],
                    hn["article_num"], hn["rerank_score"], hn["delta_rerank"],
                )

            recovered.append({
                "query"      : query,
                "golden_rank": result["golden_rank"],
                "golden_score": result["golden_score"],
                "negs_found" : neg_count,
                "best_delta" : result["best_delta"],
                "threshold"  : DELTA_THRESHOLD,
            })

            triplet = {
                "query": result["query"],
                "pos"  : result["pos"],
                "neg"  : result["neg"],
            }
            fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0

    # ── Save still-flagged ─────────────────────────────────────────────────────
    with open(STILL_FLAGGED_JSON, "w", encoding="utf-8") as f:
        json.dump(still_flagged, f, ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  STEP 1 COMPLETE")
    log.info("=" * 70)
    log.info("  Flagged queries input    : %d", total)
    log.info("  Recovered (Δ ≤ %.2f)    : %d", DELTA_THRESHOLD, len(recovered))
    log.info("  Still flagged            : %d", len(still_flagged))
    log.info("  No golden found          : %d", len(no_golden))
    log.info("  Elapsed                  : %.1f s", elapsed)
    log.info("  Triplets → %s", OUTPUT_JSONL.name)
    log.info("  Still flagged → %s", STILL_FLAGGED_JSON.name)
    log.info("=" * 70)

    # ── Per-query report table ─────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  STEP 1 — PER-QUERY RECOVERY REPORT")
    print("═" * 90)
    print(f"  {'#':<3}  {'Query':<42}  {'Status':<14}  {'GoldRank':<9}  "
          f"{'GoldScore':<10}  {'Negs':<5}  {'BestΔ'}")
    print("─" * 90)
    for rec in recovered:
        q_short = rec["query"][:40]
        print(f"  ✅   {q_short:<42}  {'recovered':<14}  "
              f"{rec['golden_rank']:<9}  {rec['golden_score']:<10.4f}  "
              f"{rec['negs_found']:<5}  {rec['best_delta']:.4f}")
    for rec in still_flagged:
        q_short = rec["query"][:40]
        print(f"  ⚑    {q_short:<42}  {'still_flagged':<14}  "
              f"{rec['golden_rank']:<9}  {rec['golden_score']:<10.4f}  "
              f"{'0':<5}  {rec['best_delta']:.4f}")
    print("═" * 90)
    print(f"  Recovered: {len(recovered)} / {total}   |   "
          f"Still flagged: {len(still_flagged)}   |   "
          f"No golden: {len(no_golden)}")
    print()


if __name__ == "__main__":
    main()
