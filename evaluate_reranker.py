#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_reranker.py  —  Script 3 / 4
========================================
Three-way comparison on the full 127-question evaluation set.

CRITICAL: NO penalties, NO score modifications, NO cross-code adjustments.
Precision@1 is purely rank-based. The only question is: what rank did
the gold article receive?

All three arms share the SAME top-30 candidate pool from BM25 + BGE-M3.

Metrics (identical formula for all three arms):
  Precision@1 = mean(1 if rank==1 else 0)
  MRR         = mean(1/rank if rank>0 else 0)
  Hit Rate@5  = mean(1 if rank<=5 else 0)
  NDCG@5      = mean(ndcg5(rank))

Output: evaluation_comparison.json  +  printed comparison table
"""

import gc
import json
import logging
import sys
import time
from math import log2
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluate_reranker.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH  = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
OUTPUT_JSON   = PROJECT_ROOT / "evaluation_comparison.json"
FT_MODEL_PATH = str(PROJECT_ROOT / "reranker_finetuned")
BGE_M3_PATH   = "D:\\pfe_baya_models\\bge-m3-unified"
GENERIC_CE    = "BAAI/bge-reranker-v2-m3"
TOP_K_POOL    = 30
TOP_K_RERANK  = 5

# Query prompt — MUST match training (Script 2)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: {query}"


# ── Shared utilities ──────────────────────────────────────────────────────────
def build_doc_text(doc: dict) -> str:
    law_name = doc.get("law_name", "")
    art_num  = doc.get("article_number", "")
    title    = doc.get("title", "")
    original = doc.get("text_original", "")
    expl     = doc.get("text_explanation", "")
    summary  = doc.get("summary", "")
    kws_raw  = doc.get("keywords", "")
    kws      = " ".join(kws_raw) if isinstance(kws_raw, list) else str(kws_raw)
    return f"[{law_name} - المادة {art_num}] {title}. {original} {expl} {summary} {kws}".strip()


import re

def is_gold_match(doc: dict, gt_articles: list) -> bool:
    c_law = str(doc.get("law_name", ""))
    c_num = str(doc.get("article_number", "")).strip()
    
    for gt in gt_articles:
        # Extract number
        match_num = re.search(r'(?:المادة|مادة)\s*(\d+)', gt)
        if not match_num:
            nums = re.findall(r'\d+', gt)
            if nums:
                gt_num = str(nums[0])
            else:
                continue
        else:
            gt_num = match_num.group(1)
            
        # Extract law
        gt_law = gt.replace("قانون", "").replace("الجزائري", "").strip()
        law_types = ["العقوبات", "المدني", "التجاري", "الاسرة", "الأسرة", "العمل", "المستهلك", "الاجراءات المدنية", "الاجراءات الجزائية", "المنافسة", "المرور", "الجنسية"]
        found_gt_law = ""
        for lt in law_types:
            if lt in gt:
                found_gt_law = lt
                if lt == "الاسرة": found_gt_law = "الأسرة"
                break
                
        if c_num == gt_num:
            if not found_gt_law or found_gt_law in c_law or c_law in found_gt_law:
                return True
    return False


def ndcg_at_5(rank: int) -> float:
    if rank == 0 or rank > 5:
        return 0.0
    return (1.0 / log2(rank + 1)) / (1.0 / log2(2))   # ideal DCG = 1/log2(2)


def compute_metrics(ranks: list) -> dict:
    """Pure rank-based metrics. No penalties, no score adjustments."""
    n = max(len(ranks), 1)
    return {
        "precision_at_1": round(sum(1 if r == 1 else 0 for r in ranks) / n, 4),
        "mrr":            round(sum(1.0 / r if r > 0 else 0.0 for r in ranks) / n, 4),
        "hit_rate_5":     round(sum(1 if 0 < r <= 5 else 0 for r in ranks) / n, 4),
        "ndcg_5":         round(sum(ndcg_at_5(r) for r in ranks) / n, 4),
    }


# ── Step 1: Build shared top-30 pool ─────────────────────────────────────────
def build_candidate_pools(questions: list) -> list:
    """Retrieve top-30 candidates for each question once, shared by all arms."""
    from bm25_rag.bm25_retriever import bm25_retrieve
    from dense_rag.bge_retriever import dense_retrieve
    from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion

    log.info("Building shared top-%d candidate pools for %d queries...",
             TOP_K_POOL, len(questions))
    pools = []
    t0    = time.time()
    for i, item in enumerate(questions, 1):
        query = item["question"]
        bm25_res  = bm25_retrieve(query, top_k=100)
        dense_res = dense_retrieve(query, top_k=100)
        fused     = reciprocal_rank_fusion(bm25_res, dense_res,
                                           bm25_weight=0.3, dense_weight=0.7)
        pools.append(fused[:TOP_K_POOL])
        if i % 25 == 0:
            log.info("  Pooled %d / %d", i, len(questions))

    log.info("  Pools ready in %.1f s", time.time() - t0)
    return pools


# ── Arm A: BGE-M3 Unified ─────────────────────────────────────────────────────
def evaluate_arm_a(questions: list, pools: list) -> list:
    """BGE-M3 Unified BGEM3FlagModel reranker — current pipeline."""
    import torch
    from FlagEmbedding import BGEM3FlagModel

    log.info("\nEvaluating Arm A (BGE-M3 Unified)...")
    log.info("Loading: %s", BGE_M3_PATH)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model   = BGEM3FlagModel(BGE_M3_PATH, use_fp16=(device == "cuda"), device=device)
    ranks   = []

    t0 = time.time()
    for i, (item, candidates) in enumerate(zip(questions, pools), 1):
        query      = item["question"]
        gt_articles = item.get("articles", [])

        if not candidates:
            ranks.append(0)
            continue

        pairs = []
        for doc in candidates:
            law_name = doc.get("law_name", "")
            art_num  = doc.get("article_number", "")
            title    = doc.get("title", "")
            original = doc.get("text_original", "")
            expl     = doc.get("text_explanation", "")
            summary  = doc.get("summary", "")
            kws      = " ".join(doc.get("keywords", [])) \
                       if isinstance(doc.get("keywords"), list) else str(doc.get("keywords", ""))
            text = f"[{law_name} - المادة {art_num}] {title}. {original} {expl} {summary} {kws}".strip()
            pairs.append([query, text])

        scores_dict = model.compute_score(pairs, max_passage_length=512,
                                          weights_for_different_modes=[0.4, 0.2, 0.4])
        scores      = scores_dict["colbert+sparse+dense"]

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        rank   = 0
        for r, (doc, _) in enumerate(ranked[:TOP_K_RERANK], 1):
            if is_gold_match(doc, gt_articles):
                rank = r
                break
        ranks.append(rank)

        if i % 25 == 0:
            log.info("  Progress: %d / %d", i, len(questions))

    log.info("  Done in %.1f s", time.time() - t0)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return ranks


# ── Score pairs with cross-encoder (used by Arms B and C) ────────────────────
def score_with_crossencoder(model, tokenizer, query: str, candidates: list,
                             device: str, batch_size: int = 16) -> list:
    """Score (query, doc) pairs. Query prompt applied. No score modification."""
    import torch

    query_prompted = QUERY_PROMPT.format(query=query)
    doc_texts = [build_doc_text(doc) for doc in candidates]
    all_scores = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(doc_texts), batch_size):
            batch_q = [query_prompted] * len(doc_texts[i:i + batch_size])
            batch_d = doc_texts[i:i + batch_size]
            enc = tokenizer(
                batch_q, batch_d,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            scores  = outputs.logits.squeeze(-1).cpu().float().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

    return all_scores


def evaluate_arm_crossencoder(arm_name: str, model_path: str,
                               questions: list, pools: list) -> list:
    """Generic reranking function for Arms B and C (cross-encoder models)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("\nEvaluating %s...", arm_name)
    log.info("Loading: %s", model_path)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=1)
    model.to(device)
    if device == "cuda":
        model.half()

    ranks = []
    t0    = time.time()

    for i, (item, candidates) in enumerate(zip(questions, pools), 1):
        query      = item["question"]
        gt_articles = item.get("articles", [])

        if not candidates:
            ranks.append(0)
            continue

        scores = score_with_crossencoder(model, tokenizer, query, candidates, device)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        rank = 0
        for r, (doc, _) in enumerate(ranked[:TOP_K_RERANK], 1):
            if is_gold_match(doc, gt_articles):
                rank = r
                break
        ranks.append(rank)

        if i % 25 == 0:
            log.info("  Progress: %d / %d", i, len(questions))

    log.info("  Done in %.1f s", time.time() - t0)

    # Free GPU memory
    del model, tokenizer
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return ranks


# ── Print comparison table ─────────────────────────────────────────────────────
def print_table(m_a: dict, m_b: dict, m_c: dict):
    def pct(v):   return f"{v * 100:.2f}%"
    def delta(a, c): sign = "+" if c >= a else "-"; return f"{sign}{abs((c - a) * 100):.2f}%"

    print("\n")
    print("  ╔═══════════════════╦══════════╦══════════╦══════════╗")
    print("  ║ Metric            ║ BGE-M3   ║ Generic  ║ FT CE    ║")
    print("  ║                   ║ Unified  ║ CE       ║          ║")
    print("  ╠═══════════════════╬══════════╬══════════╬══════════╣")
    print(f"  ║ Precision@1       ║ {pct(m_a['precision_at_1']):<8} ║ {pct(m_b['precision_at_1']):<8} ║ {pct(m_c['precision_at_1']):<8} ║")
    print(f"  ║ MRR               ║ {m_a['mrr']:<8.4f} ║ {m_b['mrr']:<8.4f} ║ {m_c['mrr']:<8.4f} ║")
    print(f"  ║ Hit Rate @5       ║ {pct(m_a['hit_rate_5']):<8} ║ {pct(m_b['hit_rate_5']):<8} ║ {pct(m_c['hit_rate_5']):<8} ║")
    print(f"  ║ NDCG@5            ║ {m_a['ndcg_5']:<8.4f} ║ {m_b['ndcg_5']:<8.4f} ║ {m_c['ndcg_5']:<8.4f} ║")
    print("  ╠═══════════════════╬══════════╬══════════╬══════════╣")
    print(f"  ║ vs BGE-M3 Unified ║  base    ║ {delta(m_a['precision_at_1'], m_b['precision_at_1']):<8} ║ {delta(m_a['precision_at_1'], m_c['precision_at_1']):<8} ║")
    print("  ╚═══════════════════╩══════════╩══════════╩══════════╝")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  evaluate_reranker.py  —  Script 3 / 4")
    log.info("  NOTE: No penalties. Pure rank-based metrics.")
    log.info("=" * 60)

    if not DATASET_PATH.exists():
        log.error("Dataset not found: %s", DATASET_PATH)
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d evaluation questions.", len(questions))

    # Step 1 — Build shared candidate pools (once for all arms)
    pools = build_candidate_pools(questions)

    # Step 2 — Evaluate all three arms sequentially to avoid VRAM conflicts
    log.info("\nStep 2/3 — Evaluating all three arms sequentially...")

    ranks_a = evaluate_arm_a(questions, pools)
    ranks_b = evaluate_arm_crossencoder("Arm B (Generic CE)", GENERIC_CE, questions, pools)
    ranks_c = evaluate_arm_crossencoder(f"Arm C (Fine-tuned CE)", FT_MODEL_PATH, questions, pools)

    # Step 3 — Compute metrics
    m_a = compute_metrics(ranks_a)
    m_b = compute_metrics(ranks_b)
    m_c = compute_metrics(ranks_c)

    log.info("\nStep 3/3 — Summary")
    print_table(m_a, m_b, m_c)

    # Per-query breakdown
    per_query = []
    for i, item in enumerate(questions):
        ra, rb, rc = ranks_a[i], ranks_b[i], ranks_c[i]
        per_query.append({
            "query":           item["question"],
            "gold_article":    (item.get("articles") or [""])[0],
            "reranker_A_rank": ra,
            "reranker_B_rank": rb,
            "reranker_C_rank": rc,
            "A_to_C_improved":  (ra != 1 and rc == 1),
            "A_to_C_regressed": (ra == 1 and rc != 1),
            "B_to_C_improved":  (rb != 1 and rc == 1),
            "B_to_C_regressed": (rb == 1 and rc != 1),
        })

    imp_a_c = sum(1 for q in per_query if q["A_to_C_improved"])
    reg_a_c = sum(1 for q in per_query if q["A_to_C_regressed"])
    unch_a_c = len(questions) - imp_a_c - reg_a_c
    imp_b_c = sum(1 for q in per_query if q["B_to_C_improved"])
    reg_b_c = sum(1 for q in per_query if q["B_to_C_regressed"])

    print(f"  Reranker C vs BGE-M3 Unified:")
    print(f"    Queries improved : {imp_a_c}")
    print(f"    Queries regressed: {reg_a_c}")
    print(f"    Queries unchanged: {unch_a_c}")
    print()

    # PASS / FAIL verdict
    ft_p1  = m_c["precision_at_1"]
    bge_p1 = m_a["precision_at_1"]
    if ft_p1 > bge_p1:
        print(f"  [PASS] Fine-tuned CE ({ft_p1*100:.2f}%) beats BGE-M3 Unified ({bge_p1*100:.2f}%) on Precision@1")
        print(f"         You may now run: python update_pipeline.py")
    else:
        print(f"  [FAIL] Fine-tuned CE ({ft_p1*100:.2f}%) does NOT beat BGE-M3 Unified ({bge_p1*100:.2f}%)")
        print(f"         Do NOT run update_pipeline.py — investigate training data.")

    # Save JSON output
    output = {
        "evaluation_note":            "No cross-code penalties applied. Pure rank-based metrics.",
        "reranker_A_bgem3_unified":   m_a,
        "reranker_B_generic_ce":      m_b,
        "reranker_C_finetuned_ce":    m_c,
        "improvements_A_to_C": {
            "precision_at_1_delta": round(ft_p1 - bge_p1, 4),
            "mrr_delta":            round(m_c["mrr"] - m_a["mrr"], 4),
            "queries_improved":     imp_a_c,
            "queries_regressed":    reg_a_c,
            "queries_unchanged":    unch_a_c,
        },
        "improvements_B_to_C": {
            "precision_at_1_delta": round(ft_p1 - m_b["precision_at_1"], 4),
            "mrr_delta":            round(m_c["mrr"] - m_b["mrr"], 4),
            "queries_improved":     imp_b_c,
            "queries_regressed":    reg_b_c,
        },
        "per_query": per_query,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("Results saved -> %s", OUTPUT_JSON.name)


if __name__ == "__main__":
    main()
