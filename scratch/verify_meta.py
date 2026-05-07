import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm25_rag.bm25_loader import load_all_articles
from dense_rag.bge_embedder import build_searchable_string

articles = load_all_articles()
with open("scratch/verify_output.txt", "w", encoding="utf-8") as out:
    for i in range(10):
        out.write(f"Article {i}:\n")
        out.write(f"  String: {build_searchable_string(articles[i])}\n")
        out.write("-" * 20 + "\n")
print("Done. Check scratch/verify_output.txt")
