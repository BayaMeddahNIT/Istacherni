import json
import re
import sys
import time
from pathlib import Path

# Add the project root to the python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from bm25_rag.bm25_retriever import bm25_retrieve
from dense_rag.bge_retriever import dense_retrieve
from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion

def extract_article_info(text):
    match_num = re.search(r'(?:المادة|مادة)\s*(\d+)', text)
    if not match_num:
        nums = re.findall(r'\d+', text)
        if nums:
            num = nums[0]
        else:
            return None, None
    else:
        num = match_num.group(1)
    
    law_types = ["العقوبات", "المدني", "التجاري", "الاسرة", "الأسرة", "العمل", "المستهلك", "الاجراءات المدنية", "الاجراءات الجزائية", "المنافسة", "المرور", "الجنسية"]
    found_law = "unknown"
    for lt in law_types:
        if lt in text:
            found_law = lt
            if lt == "الاسرة": found_law = "الأسرة"
            break
            
    return str(num), found_law

def extract_law_name(text):
    law_types = ["العقوبات", "المدني", "التجاري", "الاسرة", "الأسرة", "العمل", "المستهلك", "الاجراءات المدنية", "الاجراءات الجزائية", "المنافسة", "المرور", "الجنسية"]
    for lt in law_types:
        if lt in text:
            if lt == "الاسرة": return "الأسرة"
            return lt
    return "unknown"

def main():
    with open("algerian_law_ragas_dataset_v2.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    ground_truth_map = {}
    for item in dataset:
        q = item["question"].strip()
        gt_articles = item.get("articles", [])
        parsed_gt = []
        for art in gt_articles:
            num, law = extract_article_info(art)
            if num:
                parsed_gt.append((num, law))
        ground_truth_map[q] = parsed_gt

    questions = list(ground_truth_map.keys())
    print(f"Loaded {len(questions)} questions.")

    start_time = time.time()
    retrieval_cache = []
    
    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"Retrieving question {i+1}/{len(questions)}...", flush=True)
            
        try:
            # Fetch candidates from sparse and dense
            fetch_k = 100
            bm25_results = bm25_retrieve(q, top_k=fetch_k)
            dense_results = dense_retrieve(q, top_k=fetch_k)
            retrieval_cache.append((q, bm25_results, dense_results))
        except Exception as e:
            print(f"Error on question {i}: {e}")
            retrieval_cache.append((q, [], []))

    print(f"Retrieval finished in {time.time() - start_time:.2f} seconds.")

    # Test different weights
    weights_to_test = [
        (0.0, 1.0), # Pure Dense
        (0.3, 0.7), # Hybrid
        (0.5, 0.5)  # Equal
    ]
    
    for bm25_w, dense_w in weights_to_test:
        print(f"\n=== Testing Weights: BM25={bm25_w:.1f}, Dense={dense_w:.1f} ===")
        results = []
        for q, bm25_results, dense_results in retrieval_cache:
            # Apply RRF Fusion
            fused = reciprocal_rank_fusion(bm25_results, dense_results, bm25_weight=bm25_w, dense_weight=dense_w)
            
            # Extract retrieved article identifiers
            parsed_sources = []
            for chunk in fused:
                law_name = chunk.get("law_name", "unknown")
                article_num = str(chunk.get("article_number", "N/A"))
                
                # Check law type
                extracted_law = extract_law_name(law_name)
                if extracted_law == "unknown":
                    extracted_law = extract_law_name(chunk.get("title", law_name))
                
                parsed_sources.append((article_num, extracted_law))
                
            results.append({
                "question": q,
                "retrieved": parsed_sources
            })

        for k in [5, 35, 50, 100]:
            hit_rates = []
            for res in results:
                q = res["question"]
                gt = ground_truth_map[q]
                retrieved = res["retrieved"][:k]
                
                has_hit = False
                for r in retrieved:
                    is_relevant = False
                    for g in gt:
                        if r[0] == g[0] and (r[1] == g[1] or r[1] == "unknown" or g[1] == "unknown"):
                            is_relevant = True
                            break
                    if is_relevant:
                        has_hit = True
                        break
                hit_rates.append(1 if has_hit else 0)
                
            hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
            print(f"Hit Rate@{k}: {hit_rate:.4f} (over {len(hit_rates)} questions)")

if __name__ == "__main__":
    main()
