import os
import json
from pathlib import Path
import sys

# Fix for Windows terminal Arabic encoding
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def find_unknown_articles(data_dir):
    """
    Scans a directory for JSON files and identifies articles with 'Unknown Law' status.
    """
    print(f"Searching for unknown articles in: {data_dir}\n")
    print("-" * 100)
    print(f"{'File Path':<50} | {'Article ID/No':<20} | {'Title/Snippet'}")
    print("-" * 100)

    found_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        if isinstance(data, dict):
                            data = [data]
                            
                        for index, article in enumerate(data):
                            law_name = article.get("law_name")
                            law_domain = article.get("law_domain")
                            
                            if not law_name and not law_domain:
                                found_count += 1
                                
                                # Identify the article
                                art_id = article.get("id") or article.get("article_number") or f"Index {index}"
                                
                                # Get a title or text snippet
                                title = article.get("title") or ""
                                if not title:
                                    text = article.get("text", {})
                                    if isinstance(text, dict):
                                        text_content = text.get("original", "")
                                    else:
                                        text_content = str(text)
                                    title = text_content[:40].replace("\n", " ") + "..." if text_content else "No content"
                                
                                # Print relative path for readability
                                rel_path = os.path.relpath(file_path, data_dir)
                                print(f"{rel_path:<50} | {str(art_id):<20} | {title}")
                                
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    print("-" * 100)
    print(f"Total Unknown Articles Found: {found_count}")
    print("-" * 100)

if __name__ == "__main__":
    dataset_path = Path("dataset/raw")
    if dataset_path.exists():
        find_unknown_articles(dataset_path)
    else:
        print(f"Error: Directory '{dataset_path}' not found.")
