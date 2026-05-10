import json
import re

def normalize_law_name(text):
    text = text.replace("قانون", "").replace("الجزائري", "").strip()
    return text

def extract_article_info(text):
    # Try to find the article number
    match_num = re.search(r'(?:المادة|مادة)\s*(\d+)', text)
    if not match_num:
        # Some might just be numbers? Let's check if there are numbers
        nums = re.findall(r'\d+', text)
        if nums:
            num = nums[0]
        else:
            return None, None
    else:
        num = match_num.group(1)
    
    # Try to extract law type
    law_types = ["العقوبات", "المدني", "التجاري", "الاسرة", "الأسرة", "العمل", "المستهلك", "الاجراءات المدنية", "الاجراءات الجزائية", "المنافسة", "المرور", "الجنسية"]
    found_law = "unknown"
    for lt in law_types:
        if lt in text:
            found_law = lt
            if lt == "الاسرة": found_law = "الأسرة"
            break
            
    return str(num), found_law

def main():
    # Load ground truth
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

    # Parse answers file
    with open("answers_bge_bm25_final_1.txt", "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("==================================================")
    
    results = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Extract question
        q_match = re.search(r'\[User\]:\s*(.*?)\n', block)
        if not q_match:
            continue
        q = q_match.group(1).strip()
        
        # Extract sources
        sources_match = re.search(r'SOURCES:\n(.*)', block, re.DOTALL)
        if not sources_match:
            continue
            
        sources_text = sources_match.group(1).strip()
        source_lines = sources_text.split('\n')
        
        parsed_sources = []
        for line in source_lines:
            if not line.strip(): continue
            num, law = extract_article_info(line)
            if num:
                parsed_sources.append((num, law))
                
        results.append({
            "question": q,
            "retrieved": parsed_sources
        })

    # Calculate Precision@K and Hit Rate@K
    debug_info = []
    for k in [1, 3, 5]:
        precisions = []
        hit_rates = []
        for res in results:
            q = res["question"]
            if q not in ground_truth_map:
                continue
                
            gt = ground_truth_map[q]
            retrieved = res["retrieved"][:k]
            
            # calculate precision
            relevant_count = 0
            has_hit = False
            for r in retrieved:
                # check if r is in gt
                is_relevant = False
                for g in gt:
                    if r[0] == g[0] and (r[1] == g[1] or r[1] == "unknown" or g[1] == "unknown"):
                        is_relevant = True
                        break
                if is_relevant:
                    relevant_count += 1
                    has_hit = True
            
            precision = relevant_count / k if k > 0 else 0
            precisions.append(precision)
            hit_rates.append(1 if has_hit else 0)
            
            if k == 1:
                debug_info.append({
                    "question": q,
                    "gt": gt,
                    "retrieved": retrieved,
                    "is_relevant": is_relevant if retrieved else False
                })
            
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        print(f"k={k} | Precision@{k}: {avg_precision:.4f} | Hit Rate@{k}: {hit_rate:.4f} (over {len(precisions)} questions)")
        
    with open("precision_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
