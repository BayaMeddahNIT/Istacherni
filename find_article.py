import os
import json
import sys
from pathlib import Path

# Fix for Windows terminal Arabic encoding
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def find_articles_by_number(article_num, data_dir="dataset/raw"):
    """
    Searches for articles with a specific number across all JSON files in the directory.
    """
    found_articles = []
    
    # Path to the raw dataset
    path = Path(data_dir)
    if not path.exists():
        # Fallback to current directory or ignore if searching in raw is not possible
        return []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        # Handle both single dict or list of dicts
                        if isinstance(data, dict):
                            data = [data]
                            
                        for article in data:
                            # Check if 'article_number' matches
                            if str(article.get("article_number")) == str(article_num):
                                found_articles.append({
                                    "law": article.get("law_name", "Unknown Law"),
                                    "number": article.get("article_number"),
                                    "title": article.get("title", "No Title"),
                                    "text": article.get("text", {}).get("original", "No Text"),
                                    "file": file
                                })
                except Exception:
                    pass 

    return found_articles

def find_related_questions(article_num, rag_file="algerian_law_ragas_dataset_v3.json"):
    """
    Searches for questions that mention the article number in the RAG dataset.
    """
    found_questions = []
    rag_path = Path(rag_file)
    if not rag_path.exists():
        return []

    try:
        with open(rag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # The search pattern for article number in the dataset
            # e.g., "المادة 219"
            search_pattern = f"المادة {article_num}"
            
            for item in data:
                articles_list = item.get("articles", [])
                # Check if any article in the list contains "المادة X"
                if any(search_pattern in art for art in articles_list):
                    found_questions.append({
                        "question": item.get("question"),
                        "matching_articles": [art for art in articles_list if search_pattern in art]
                    })
    except Exception:
        pass

    return found_questions

def main():
    print("=== Algerian Law Article & Question Finder ===")
    print("(Type 'exit', 'quit', or 'q' to stop)")
    
    while True:
        try:
            print("\n" + "#" * 60)
            user_input = input("Enter the article number: ").strip().lower()
            
            if user_input in ['exit', 'quit', 'q']:
                print("Exiting... Goodbye!")
                break
                
            if not user_input:
                continue

            # 1. Find Raw Articles
            articles = find_articles_by_number(user_input)
            
            # 2. Find Related Questions from RAG Dataset
            questions = find_related_questions(user_input)

            print("=" * 60)
            if not articles:
                print(f"No raw articles found with number {user_input}.")
            else:
                print(f"Found {len(articles)} raw article(s) with number {user_input}:\n")
                for idx, art in enumerate(articles, 1):
                    print(f"{idx}. Law: {art['law']}")
                    print(f"   Article: {art['number']}")
                    print(f"   Title: {art['title']}")
                    # Improved display for Arabic text
                    text_display = art['text'].replace('\n', ' ')
                    print(f"   Content: {text_display[:500]}..." if len(text_display) > 500 else f"   Content: {text_display}")
                    print("-" * 50)

            print("\n" + "=" * 60)
            if not questions:
                print(f"No related questions found in RAG dataset for article {user_input}.")
            else:
                print(f"Found {len(questions)} related question(s) in RAG dataset:\n")
                for idx, q in enumerate(questions, 1):
                    print(f"{idx}. Question: {q['question']}")
                    print(f"   Referenced Articles: {', '.join(q['matching_articles'])}")
                    print("-" * 50)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
