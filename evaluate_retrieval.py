import json
import time
import sys
import csv
from pathlib import Path

# Add the project root to the python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError in Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("Initializing Hybrid Retrieval Evaluation... Please wait while models load.", flush=True)

from hybrid_rag.hybrid_retriever import hybrid_retrieve

DATASET_PATH = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
OUTPUT_CSV = PROJECT_ROOT / "retrieval_evaluation_results.csv"

def match_articles(gt_articles_list, retrieved_chunks):
    """
    Checks how many ground-truth articles were successfully retrieved.
    
    gt_articles_list: ["قانون العقوبات - المادة 219", "القانون التجاري - المادة 28", ...]
    retrieved_chunks: List of dictionaries returned by hybrid_retrieve
    
    Returns:
        matched_count (int): Number of ground truth articles found in the retrieved chunks.
        is_hit (bool): True if at least one ground truth article was found.
        matched_details (list): Which exact articles were found.
        rank_of_first_hit (int): Rank of the first matched article (1-indexed), or 0 if no hit.
    """
    gt_matched = set()
    rank_of_first_hit = 0
    
    # Parse ground truth articles
    parsed_gts = []
    for gt in gt_articles_list:
        parts = gt.split(" - المادة ")
        if len(parts) == 2:
            gt_law = parts[0].strip()
            gt_num = parts[1].strip()
        else:
            gt_law = ""
            gt_num = "".join(filter(str.isdigit, gt))
        parsed_gts.append({"raw": gt, "law": gt_law, "num": gt_num})
            
    for rank, chunk in enumerate(retrieved_chunks, 1):
        c_law = str(chunk.get("law_name", ""))
        c_num = str(chunk.get("article_number", ""))
        
        chunk_matched = False
        for gt in parsed_gts:
            if c_num == gt["num"]:
                if not gt["law"] or gt["law"] in c_law or c_law in gt["law"] or ("قانون" in gt["law"] and "قانون" in c_law):
                    gt_matched.add(gt["raw"])
                    chunk_matched = True
        
        if chunk_matched and rank_of_first_hit == 0:
            rank_of_first_hit = rank
                    
    return len(gt_matched), len(gt_matched) > 0, list(gt_matched), rank_of_first_hit

def main():
    if not DATASET_PATH.exists():
        print(f"Error: Dataset {DATASET_PATH} not found.")
        return
        
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH.name}")
    print("=" * 60)
    
    total_questions = len(dataset)
    total_hits = 0
    total_recall = 0.0
    total_reciprocal_rank = 0.0
    total_precision_at_1 = 0
    
    results_records = []
    
    start_time = time.time()
    
    for i, item in enumerate(dataset, 1):
        question = item.get("question", "")
        gt_articles = item.get("articles", [])
        
        print(f"[{i}/{total_questions}] Q: {question}")
        
        if not gt_articles:
            print("  -> Warning: No ground truth articles for this question. Skipping evaluation.")
            continue
            
        try:
            # Retrieve top 5 chunks
            chunks = hybrid_retrieve(question, top_k=5)
            
            # Match chunks against ground truth
            matched_count, is_hit, matched_details, rank_of_first_hit = match_articles(gt_articles, chunks)
            
            # Calculate metrics for this question
            recall = matched_count / len(gt_articles)
            
            reciprocal_rank = 0.0
            precision_at_1 = 0
            
            if is_hit:
                total_hits += 1
                reciprocal_rank = 1.0 / rank_of_first_hit
                if rank_of_first_hit == 1:
                    precision_at_1 = 1
                    total_precision_at_1 += 1
                    
            total_recall += recall
            total_reciprocal_rank += reciprocal_rank
            
            print(f"  -> Expected: {len(gt_articles)} | Found: {matched_count} | Recall@5: {recall:.2f} | Rank: {rank_of_first_hit}")
            if is_hit:
                print(f"  -> Matched: {', '.join(matched_details)}")
                
            # Log results
            results_records.append({
                "question": question,
                "expected_articles": " | ".join(gt_articles),
                "retrieved_articles": " | ".join([f"{c.get('law_name')} - المادة {c.get('article_number')}" for c in chunks]),
                "matched_count": matched_count,
                "total_expected": len(gt_articles),
                "is_hit": is_hit,
                "recall_at_5": round(recall, 4),
                "rank_of_first_hit": rank_of_first_hit,
                "reciprocal_rank": round(reciprocal_rank, 4),
                "precision_at_1": precision_at_1
            })
            
        except Exception as e:
            print(f"  -> Error retrieving for this question: {e}")
            
    elapsed = time.time() - start_time
    
    # Calculate final aggregate metrics
    num_evaluated = len(results_records)
    final_hit_rate = total_hits / num_evaluated if num_evaluated else 0
    final_recall = total_recall / num_evaluated if num_evaluated else 0
    final_mrr = total_reciprocal_rank / num_evaluated if num_evaluated else 0
    final_precision_at_1 = total_precision_at_1 / num_evaluated if num_evaluated else 0
    
    # Write CSV
    if results_records:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "expected_articles", "retrieved_articles", 
                "matched_count", "total_expected", "is_hit", "recall_at_5",
                "rank_of_first_hit", "reciprocal_rank", "precision_at_1"
            ])
            writer.writeheader()
            for row in results_records:
                writer.writerow(row)
                
    # Print Summary
    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION SUMMARY (Hybrid BGE-M3 + BM25)")
    print("=" * 60)
    print(f"Total Questions Evaluated : {num_evaluated}")
    print(f"MRR (Mean Reciprocal Rank): {final_mrr:.4f}")
    print(f"Precision@1 (Top-1 Acc)   : {final_precision_at_1:.2%}")
    print(f"Hit Rate (Top 5)          : {final_hit_rate:.2%}")
    print(f"Average Recall@5          : {final_recall:.2%}")
    print(f"Total Time Taken          : {elapsed:.2f} seconds")
    print(f"Results saved to          : {OUTPUT_CSV.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
