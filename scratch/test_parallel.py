import sys
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_rag.bge_embedder import build_searchable_string, _ollama_embed

def embed_articles_parallel(articles, max_workers=8):
    texts = [build_searchable_string(a) for a in articles]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_ollama_embed, texts))
    return results

if __name__ == "__main__":
    from bm25_rag.bm25_loader import load_all_articles
    articles = load_all_articles()[:10]
    print(f"Testing parallel embedding of {len(articles)} articles...")
    res = embed_articles_parallel(articles)
    print(f"Done. Result length: {len(res)}")
