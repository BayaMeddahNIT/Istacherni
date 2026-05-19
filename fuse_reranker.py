#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import json
import logging
import sys
import time
import torch
from math import log2
from pathlib import Path

# Setup logging
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fuse_reranker.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_reranker import build_candidate_pools, is_gold_match, build_doc_text, ndcg_at_5, compute_metrics, QUERY_PROMPT

DATASET_PATH = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
BGE_M3_PATH = "D:\\pfe_baya_models\\bge-m3-unified"
FINETUNED_CE_PATH = PROJECT_ROOT / "reranker_finetuned"
TOP_K_RERANK = 5

def normalize(scores):
    if not scores: return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def score_bgem3(questions, pools):
    from FlagEmbedding import BGEM3FlagModel
    log.info("Loading BGE-M3 Unified...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(BGE_M3_PATH, use_fp16=(device=="cuda"), device=device)
    
    all_scores = []
    for i, (item, candidates) in enumerate(zip(questions, pools)):
        if not candidates:
            all_scores.append([])
            continue
            
        query = item["question"]
        pairs = []
        for doc in candidates:
            text = build_doc_text(doc)
            pairs.append([query, text])
            
        scores_dict = model.compute_score(pairs, max_passage_length=512, weights_for_different_modes=[0.4, 0.2, 0.4])
        scores = scores_dict["colbert+sparse+dense"]
        all_scores.append(scores)
        
        if (i+1) % 25 == 0: log.info(f"BGE-M3 scored {i+1} / {len(questions)}")
        
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return all_scores

def score_finetuned_ce(questions, pools):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    log.info("Loading Fine-Tuned CE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_CE_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_CE_PATH)).to(device)
    model.eval()
    if device == "cuda": model.half()
    
    all_scores = []
    with torch.no_grad():
        for i, (item, candidates) in enumerate(zip(questions, pools)):
            if not candidates:
                all_scores.append([])
                continue
                
            query = item["question"]
            query_prompted = QUERY_PROMPT.format(query=query)
            doc_texts = [build_doc_text(doc) for doc in candidates]
            
            # Batch process
            batch_size = 16
            query_scores = []
            for j in range(0, len(doc_texts), batch_size):
                batch_q = [query_prompted] * len(doc_texts[j:j+batch_size])
                batch_d = doc_texts[j:j+batch_size]
                enc = tokenizer(batch_q, batch_d, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
                outputs = model(**enc)
                logits = outputs.logits.squeeze(-1)
                if logits.dim() == 0: logits = logits.unsqueeze(0)
                scores = torch.sigmoid(logits.float()).cpu().numpy().tolist()
                query_scores.extend(scores)
            all_scores.append(query_scores)
            
            if (i+1) % 25 == 0: log.info(f"CE scored {i+1} / {len(questions)}")
            
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return all_scores

def main():
    log.info("Loading questions...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    pools = build_candidate_pools(questions)
    
    scores_A = score_bgem3(questions, pools)
    scores_B = score_finetuned_ce(questions, pools)
    
    # Normalization & Sweep
    log.info("Sweeping alphas...")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results = []
    
    # Pre-normalize
    norm_A = [normalize(s) for s in scores_A]
    norm_B = [normalize(s) for s in scores_B]
    
    best_alpha = -1
    best_p1 = -1
    best_mrr = -1
    baseline_p1 = -1
    ftce_p1 = -1
    best_hit = -1
    best_ndcg = -1
    
    for alpha in alphas:
        ranks = []
        for i, (item, candidates) in enumerate(zip(questions, pools)):
            if not candidates:
                ranks.append(0)
                continue
                
            gt_articles = item.get("articles", [])
            nA = norm_A[i]
            nB = norm_B[i]
            
            fused = [alpha * a + (1 - alpha) * b for a, b in zip(nA, nB)]
            ranked = sorted(zip(candidates, fused), key=lambda x: x[1], reverse=True)
            
            rank = 0
            for r, (doc, _) in enumerate(ranked[:TOP_K_RERANK], 1):
                if is_gold_match(doc, gt_articles):
                    rank = r
                    break
            ranks.append(rank)
            
        metrics = compute_metrics(ranks)
        p1 = metrics["precision_at_1"]
        mrr = metrics["mrr"]
        hit = metrics["hit_rate_5"]
        ndcg = metrics["ndcg_5"]
        
        sweep_results.append({
            "alpha": alpha,
            "precision_at_1": p1,
            "mrr": mrr,
            "hit_rate_5": hit,
            "ndcg_5": ndcg
        })
        
        if alpha == 1.0: baseline_p1 = p1
        if alpha == 0.0: ftce_p1 = p1
        
        if p1 > best_p1 or (p1 == best_p1 and mrr > best_mrr):
            best_alpha = alpha
            best_p1 = p1
            best_mrr = mrr
            best_hit = hit
            best_ndcg = ndcg

    # Per-query analysis at best alpha
    per_query = []
    for i, (item, candidates) in enumerate(zip(questions, pools)):
        if not candidates: continue
        gt_articles = item.get("articles", [])
        nA = norm_A[i]
        nB = norm_B[i]
        
        def get_rank(scores):
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            for r, (doc, _) in enumerate(ranked[:TOP_K_RERANK], 1):
                if is_gold_match(doc, gt_articles): return r
            return 0
            
        rank_A = get_rank(nA)
        rank_B = get_rank(nB)
        
        fused = [best_alpha * a + (1 - best_alpha) * b for a, b in zip(nA, nB)]
        rank_fused = get_rank(fused)
        
        per_query.append({
            "query": item["question"],
            "gold_articles": gt_articles,
            "bgem3_rank": rank_A,
            "ftce_rank": rank_B,
            "fused_rank": rank_fused,
            "fused_improved_vs_bgem3": (rank_A != 1 and rank_fused == 1),
            "fused_regressed_vs_bgem3": (rank_A == 1 and rank_fused != 1),
            "score_A_normalized": nA,
            "score_B_normalized": nB,
            "fused_score": fused
        })
        
    print("\nAlpha | P@1    | MRR    | vs BGE-M3 Unified")
    print("-" * 50)
    for res in sweep_results:
        vs_base = res["precision_at_1"] - baseline_p1
        sign = "+" if vs_base >= 0 else ""
        print(f"{res['alpha']:<5.1f} | {res['precision_at_1']*100:>5.2f}% | {res['mrr']:.4f} | {sign}{vs_base*100:.2f}%")

    out_sweep = {
        "sweep": sweep_results,
        "best_alpha": best_alpha,
        "best_precision_at_1": best_p1,
        "best_mrr": best_mrr,
        "baseline_bgem3_precision_at_1": baseline_p1,
        "baseline_ftce_precision_at_1": ftce_p1,
        "improvement_over_bgem3": round(best_p1 - baseline_p1, 4)
    }
    
    with open("alpha_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(out_sweep, f, indent=2, ensure_ascii=False)
        
    with open("fusion_per_query.json", "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)
        
    log.info("Saved alpha_sweep_results.json and fusion_per_query.json")
    
if __name__ == "__main__":
    main()
