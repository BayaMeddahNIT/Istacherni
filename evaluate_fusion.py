#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_fusion.py  —  Script 3 / 3
======================================
Final verification comparing the end-to-end pipeline against the sweeps.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluate_fusion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_reranker import build_candidate_pools, is_gold_match, compute_metrics
DATASET_PATH  = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
SWEEP_RESULTS = PROJECT_ROOT / "alpha_sweep_results.json"

def main():
    if not SWEEP_RESULTS.exists():
        log.error("alpha_sweep_results.json not found.")
        sys.exit(1)

    with open(SWEEP_RESULTS, "r", encoding="utf-8") as f:
        sweep_data = json.load(f)

    sweep = sweep_data["sweep"]
    best_alpha = sweep_data["best_alpha"]

    m_A = next(x for x in sweep if x["alpha"] == 1.0)
    m_B = next(x for x in sweep if x["alpha"] == 0.0)
    m_C = next(x for x in sweep if x["alpha"] == best_alpha)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    pools = build_candidate_pools(questions)

    log.info("\nEvaluating Arm D: End-to-End Pipeline Check (rerank_candidates)")
    try:
        from hybrid_rag.reranker import rerank_candidates
    except ImportError:
        log.error("Failed to import rerank_candidates. Did you run update_reranker_fused.py?")
        sys.exit(1)

    ranks_D = []
    t0 = time.time()
    for i, (item, candidates) in enumerate(zip(questions, pools), 1):
        if not candidates:
            ranks_D.append(0)
            continue
            
        query = item["question"]
        gt_articles = item.get("articles", [])
        
        reranked = rerank_candidates(query, candidates, top_k=5)
        
        rank = 0
        for r, doc in enumerate(reranked, 1):
            if is_gold_match(doc, gt_articles):
                rank = r
                break
        ranks_D.append(rank)

        if i % 25 == 0:
            log.info("  Progress: %d / %d", i, len(questions))

    log.info("  Done in %.1f s", time.time() - t0)
    m_D = compute_metrics(ranks_D)

    def pct(v): return f"{v * 100:.2f}%"
    def diff(v, base): 
        d = v - base
        return f"{'+' if d>=0 else ''}{d*100:.2f}%"

    base_p1 = m_A['precision_at_1']
    
    print("\n")
    print("  ╔═══════════════╦══════════╦══════════╦══════════╦══════════╗")
    print("  ║ Metric        ║ BGE-M3   ║ FT CE    ║ Fused    ║ Pipeline ║")
    print(f"  ║               ║ Unified  ║ Only     ║ α={best_alpha:<4.1f}   ║ Check    ║")
    print("  ╠═══════════════╬══════════╬══════════╬══════════╬══════════╣")
    print(f"  ║ Precision@1   ║ {pct(m_A['precision_at_1']):<8} ║ {pct(m_B['precision_at_1']):<8} ║ {pct(m_C['precision_at_1']):<8} ║ {pct(m_D['precision_at_1']):<8} ║")
    print(f"  ║ MRR           ║ {m_A['mrr']:<8.4f} ║ {m_B['mrr']:<8.4f} ║ {m_C['mrr']:<8.4f} ║ {m_D['mrr']:<8.4f} ║")
    print(f"  ║ Hit Rate @5   ║ {pct(m_A['hit_rate_5']):<8} ║ {pct(m_B['hit_rate_5']):<8} ║ {pct(m_C['hit_rate_5']):<8} ║ {pct(m_D['hit_rate_5']):<8} ║")
    print(f"  ║ NDCG@5        ║ {m_A['ndcg_5']:<8.4f} ║ {m_B['ndcg_5']:<8.4f} ║ {m_C['ndcg_5']:<8.4f} ║ {m_D['ndcg_5']:<8.4f} ║")
    print("  ╠═══════════════╬══════════╬══════════╬══════════╬══════════╣")
    print(f"  ║ vs BGE-M3     ║  base    ║ {diff(m_B['precision_at_1'], base_p1):<8} ║ {diff(m_C['precision_at_1'], base_p1):<8} ║ {diff(m_D['precision_at_1'], base_p1):<8} ║")
    print("  ╚═══════════════╩══════════╩══════════╩══════════╩══════════╝\n")

    if m_D['precision_at_1'] > base_p1:
        print("  [PASS] Fusion improves over baseline.")
        print(f"         Pipeline correctly updated with FUSION_ALPHA={best_alpha}")
    else:
        print("  [HOLD] Fusion did not improve P@1.")
        print("         Keep BGE-M3 Unified only (USE_FUSION=false)")
        if m_D['mrr'] > m_A['mrr']: print("         (Note: MRR improved)")
        if m_D['hit_rate_5'] > m_A['hit_rate_5']: print("         (Note: Hit Rate improved)")
        if m_D['ndcg_5'] > m_A['ndcg_5']: print("         (Note: NDCG improved)")

if __name__ == "__main__":
    main()
