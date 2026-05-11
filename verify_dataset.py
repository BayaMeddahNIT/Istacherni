import os
import json
import re
import sys
from pathlib import Path

# Fix for Windows terminal Arabic encoding
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def parse_rag_article(article_str):
    """
    Parses a string like 'قانون العقوبات - المادة 219' into (law_name, article_number).
    """
    # Remove any extra spaces
    article_str = article_str.strip()
    
    # Pattern to match 'Law Name - المادة Number'
    # Supporting both '-' and '—' and other separators if any
    parts = re.split(r'[-\u2014]', article_str)
    
    if len(parts) >= 2:
        law_name = parts[0].strip()
        article_part = parts[1].strip()
        
        # Extract numbers from the article part (e.g., 'المادة 219' -> '219')
        numbers = re.findall(r'\d+', article_part)
        if numbers:
            return law_name, numbers[0]
            
    return None, None

def get_raw_articles_index(data_dir="dataset/raw"):
    """
    Returns a set of (law_name, article_number) tuples from the raw dataset.
    """
    raw_index = set()
    path = Path(data_dir)
    if not path.exists():
        return raw_index

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            data = [data]
                        for article in data:
                            law = article.get("law_name")
                            num = article.get("article_number")
                            if law and num is not None:
                                # Normalize law name and convert num to string
                                raw_index.add((law.strip(), str(num)))
                except Exception:
                    pass
    return raw_index

def verify_dataset(rag_file="algerian_law_ragas_dataset_v3.json", raw_dir="dataset/raw"):
    print(f"Loading RAG dataset: {rag_file}")
    try:
        with open(rag_file, "r", encoding="utf-8") as f:
            rag_data = json.load(f)
    except Exception as e:
        print(f"Error loading {rag_file}: {e}")
        return

    print(f"Indexing raw articles from: {raw_dir}")
    raw_index = get_raw_articles_index(raw_dir)
    print(f"Found {len(raw_index)} unique articles in raw dataset.")

    missing_articles = {}
    found_count = 0
    total_refs = 0

    # Iterate through questions
    for item in rag_data:
        articles_list = item.get("articles", [])
        for art_str in articles_list:
            total_refs += 1
            law, num = parse_rag_article(art_str)
            
            if law and num:
                # Try to find a match. Law name might be slightly different, so we check for substring or exact
                # However, for efficiency, let's try exact first.
                if (law, num) in raw_index:
                    found_count += 1
                else:
                    # Collect missing
                    if art_str not in missing_articles:
                        missing_articles[art_str] = {
                            "law": law,
                            "number": num,
                            "questions": []
                        }
                    missing_articles[art_str]["questions"].append(item.get("question"))
            else:
                # If we couldn't parse it, mark it as invalid/missing
                if art_str not in missing_articles:
                    missing_articles[art_str] = {
                        "law": "PARSE ERROR",
                        "number": "N/A",
                        "questions": []
                    }
                missing_articles[art_str]["questions"].append(item.get("question"))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total article references in RAG dataset: {total_refs}")
    print(f"Successfully found in raw data: {found_count}")
    print(f"Missing articles: {len(missing_articles)}")
    print("="*60)

    if missing_articles:
        print("\nLIST OF MISSING ARTICLES (Please add these to the raw dataset):")
        print(f"{'Article Reference':<40} | {'Law Detected':<20} | {'Num':<5}")
        print("-" * 75)
        for art_str, info in sorted(missing_articles.items()):
            print(f"{art_str[:38]:<40} | {info['law'][:18]:<20} | {info['number']:<5}")
            
        # Optional: Save to file
        with open("missing_articles_report.json", "w", encoding="utf-8") as f:
            json.dump(missing_articles, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: missing_articles_report.json")
    else:
        print("\nGreat! All articles referenced in the RAG dataset exist in the raw data.")

if __name__ == "__main__":
    verify_dataset()
