import csv
import sys
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding="utf-8")

EVAL_CSV = Path("retrieval_evaluation_results.csv")

def _unquote(val: str) -> str:
    v = val.strip()
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        v = v[1:-1]
    return v.strip()

def main():
    questions = []
    with open(EVAL_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: _unquote(v) for k, v in row.items()}
            q   = row.get("question", "").strip()
            exp = row.get("expected_articles", "")
            gt  = [a.strip() for a in exp.split("|") if a.strip()]
            if q and gt:
                questions.append({"query": q, "gt": gt})
                
    if not questions:
        log.warning("No questions found! Check CSV parsing.")
        return

    from bm25_rag.bm25_retriever import bm25_retrieve
    from dense_rag.bge_retriever import dense_retrieve
    from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion
    from hybrid_rag.reranker import rerank_candidates

    def retrieve_fn(query: str) -> list:
        bm25_results  = bm25_retrieve(query, top_k=200)
        dense_results = dense_retrieve(query, top_k=200)
        fused = reciprocal_rank_fusion(bm25_results, dense_results, bm25_weight=0.3, dense_weight=0.7)
        return rerank_candidates(query, fused[:200], top_k=30)

    def is_golden_match(doc: dict, gt_articles: list) -> bool:
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
                if (not gt_law or gt_law in c_law or c_law in gt_law or ("قانون" in gt_law and "قانون" in c_law)):
                    return True
        return False

    log.info(f"Evaluating {len(questions)} questions...")
    precision_at_1 = []
    reciprocal_ranks = []
    
    for i, item in enumerate(questions, 1):
        query = item["query"]
        gt = item["gt"]
        try:
            results = retrieve_fn(query)
        except Exception as exc:
            precision_at_1.append(0)
            reciprocal_ranks.append(0)
            continue
            
        p1 = 1 if results and is_golden_match(results[0], gt) else 0
        precision_at_1.append(p1)
        
        hit = 0
        first_hit_rank = None
        for rank, doc in enumerate(results, 1):
            if is_golden_match(doc, gt):
                if first_hit_rank is None:
                    first_hit_rank = rank
                    
        rr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        reciprocal_ranks.append(rr)
        
        if i % 10 == 0:
            log.info(f"  Processed {i}/{len(questions)}...")

    metrics = {
        "precision_at_1": float(np.mean(precision_at_1)),
        "mrr": float(np.mean(reciprocal_ranks))
    }
    
    log.info("\n  ── POST-TRAINING METRICS ──────────────────────────────")
    log.info(f"  Precision@1  : {metrics['precision_at_1']:.4f}  (baseline: 0.9134)")
    log.info(f"  MRR          : {metrics['mrr']:.4f}  (baseline: 0.9554)")
    log.info("  ──────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
