import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from bm25_rag.bm25_loader import load_all_articles

articles = load_all_articles()
print(f"Total articles: {len(articles)}")

last_30 = articles[-30:]
for i, a in enumerate(last_30):
    try:
        # Use simple prints to avoid encoding issues or just print IDs
        print(f"{i+1}. ID: {a.get('id')} | Num: {a.get('article_number')}")
    except Exception as e:
        print(f"Error printing article {i}: {e}")
