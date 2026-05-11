# dense_rag/bge_embedder.py
import sys
import os
import json
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path("D:\\pfe_baya_models\\bge-m3-unified")

# We use sentence-transformers to load our newly fine-tuned model
import torch
from sentence_transformers import SentenceTransformer

# Global model instance
_model = None

def get_model():
    global _model
    if _model is None:
        # Use CPU to save VRAM for reranker
        device = "cpu"
        if MODEL_PATH.exists():
            _model = SentenceTransformer(str(MODEL_PATH), device=device)
        else:
            print("Warning: Fine-tuned model not found. Falling back to base BAAI/bge-m3")
            _model = SentenceTransformer("BAAI/bge-m3", device=device)
    return _model

def _ollama_embed(text: str) -> List[float]:
    """Legacy wrapper, now uses local model."""
    model = get_model()
    # Ensure float lists
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def embed_articles_batch(articles: List[dict], batch_size: int = 16) -> List[List[float]]:
    """Batch embed articles efficiently."""
    texts = [build_searchable_string(a) for a in articles]
    model = get_model()
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()

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
    """Embed article using local finetuned model."""
    text = build_searchable_string(article)
    return _ollama_embed(text)

def embed_query(query: str) -> List[float]:
    """Embed query using local finetuned model."""
    # BGE-M3 likes a prefix for queries
    prefixed = f"Represent this query for retrieving relevant documents: {query}"
    return _ollama_embed(prefixed)