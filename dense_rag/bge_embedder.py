# dense_rag/bge_embedder.py
import sys
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List

# ── Config (Use Ollama for embeddings to avoid library crashes) ────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str     = "bge-m3" # You already have this in Ollama

from concurrent.futures import ThreadPoolExecutor

def _ollama_embed(text: str) -> List[float]:
    """Get embeddings from local Ollama API."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": text
    }).encode("utf-8")
    
    headers = {"Content-Type": "application/json"}
    
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]
    except Exception as e:
        print(f"[Ollama-Embed] Error: {e}", flush=True)
        return [0.0] * 1024 

def embed_articles_batch(articles: List[dict], max_workers: int = 10) -> List[List[float]]:
    """Batch embed articles in parallel for much better performance."""
    texts = [build_searchable_string(a) for a in articles]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use map to keep results in the same order as input
        results = list(executor.map(_ollama_embed, texts))
    return results

def _safe_str(value) -> str:
    if value is None: return ""
    if isinstance(value, list): return " ".join(str(v) for v in value if v is not None)
    return str(value)

def build_searchable_string(article: dict) -> str:
    law_name    = _safe_str(article.get("law_name"))
    art_num     = _safe_str(article.get("article_number"))
    title       = _safe_str(article.get("title"))
    keywords    = _safe_str(article.get("keywords"))
    original    = _safe_str(article.get("text_original"))
    explanation = _safe_str(article.get("text_explanation"))
    summary     = _safe_str(article.get("summary"))
    
    header = ""
    if law_name and art_num:
        header = f"[{law_name} - المادة {art_num}]"
    
    # Include EVERYTHING to maximize recall
    parts = [p for p in [header, title, keywords, original, explanation, summary] if p]
    return " | ".join(parts).strip() or "unknown"

def embed_article(article: dict) -> List[float]:
    """Embed article using Ollama."""
    text = build_searchable_string(article)
    return _ollama_embed(text)

def embed_query(query: str) -> List[float]:
    """Embed query using Ollama."""
    # BGE-M3 likes a prefix for queries
    prefixed = f"Represent this query for retrieving relevant documents: {query}"
    return _ollama_embed(prefixed)