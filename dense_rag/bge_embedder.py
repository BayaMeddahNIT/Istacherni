# dense_rag/bge_embedder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "BAAI/bge-m3"
_model = None
_model_loading = False  # prevent re-entry on error


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BGE-M3] Loading model on {device}…")
        _model = SentenceTransformer(MODEL_NAME, device=device)
        print("[BGE-M3] Model loaded ✓")
    return _model


def _safe_str(value) -> str:
    """Convert any value to string safely, handling None and lists."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)
    return str(value)


def build_searchable_string(article: dict) -> str:
    """
    Build the 'Searchable String' that BGE-M3 encodes at index time.

    Concatenation order:  law_name | title | keywords | text_explanation
    - law_name        → helps filter by domain (مدني / جزائي / تجاري…)
    - title           → concise topic signal
    - keywords        → both technical + common-language terms
    - text_explanation → plain-Arabic explanation (richer than the raw legal text)
    """
    law_name    = _safe_str(article.get("law_name"))
    title       = _safe_str(article.get("title"))
    keywords    = _safe_str(article.get("keywords"))   # list → space-joined
    # text_explanation is now always present after bm25_loader normalisation;
    # fall back to summary → text_original as graceful degradation
    explanation = (
        _safe_str(article.get("text_explanation"))
        or _safe_str(article.get("summary"))
        or _safe_str(article.get("text_original"))
    )

    parts = [p for p in [law_name, title, keywords, explanation] if p]
    text  = " | ".join(parts).strip()

    # Ultimate fallback
    if not text:
        text = _safe_str(article.get("id", "unknown article"))

    return text


def embed_article(article: dict) -> list[float]:
    """Embed article using the structured Searchable String for BGE-M3."""
    text  = build_searchable_string(article)
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    model = get_model()
    #prefixed = f"Represent this sentence: {query}"
    prefixed = f"Represent this query for retrieving relevant documents: {query}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()