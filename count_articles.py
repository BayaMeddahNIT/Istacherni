import os
import json
import sys
from pathlib import Path
from collections import Counter

# Fix for Windows terminal Arabic encoding
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def count_law_articles(data_dir):
    """
    Scans a directory for JSON files and counts articles by 'law_name'.
    """
    law_counts = Counter()
    total_files = 0
    total_articles = 0

    print(f"Scanning directory: {data_dir}\n")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = Path(root) / file
                total_files += 1
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        # Handle both single dict or list of dicts
                        if isinstance(data, dict):
                            data = [data]
                            
                        for article in data:
                            # Use 'law_name' as the primary key, fallback to 'law_domain'
                            law_name = article.get("law_name") or article.get("law_domain") or "Unknown Law"
                            law_counts[law_name] += 1
                            total_articles += 1
                            
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # Output results
    print("-" * 50)
    print(f"{'Law Name':<40} | {'Articles':<10}")
    print("-" * 50)
    
    # Sort by count descending
    for law, count in law_counts.most_common():
        print(f"{law:<40} | {count:<10}")
        
    print("-" * 50)
    print(f"Total Files Scanned: {total_files}")
    print(f"Total Articles Found: {total_articles}")
    print("-" * 50)

if __name__ == "__main__":
    # Point to your raw dataset directory
    dataset_path = Path("dataset/raw")
    if dataset_path.exists():
        count_law_articles(dataset_path)
    else:
        print(f"Error: Directory '{dataset_path}' not found.")
