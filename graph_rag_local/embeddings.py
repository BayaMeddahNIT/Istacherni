"""
embeddings.py
-------------
Local embedding utility for the Algerian Law Knowledge Graph.
Uses BAAI/bge-m3 via sentence-transformers.
"""

# ── Offline Enforcement ────────────────────────────────────────────────────────
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from pathlib import Path
from typing import List, Union
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# IMPORTANT: This MUST match the model used when building the vector index.
# If you change this model, you MUST delete and rebuild the entire graph index.
MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lazy-initialised model ───────────────────────────────────────────────────
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {MODEL_NAME} on {DEVICE}...")
        # Note: BGE-M3 and Qwen embeddings can be quite heavy. We use sentence-transformers for efficiency.
        # Ensure trust_remote_code=True for newer or custom architectures like Qwen
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
        print("[Embeddings] Model loaded.")
    return _model

def embed_text(text: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
    """
    Generate embeddings for a string or a list of strings.
    Returns: A numpy array of embeddings.
    """
    model = get_model()
    # BGE-M3 recommendation: use normalize_embeddings=True for better cosine similarity
    embeddings = model.encode(
        text, 
        normalize_embeddings=True, 
        show_progress_bar=show_progress,
        batch_size=16,          # Balanced for CPU performance and RAM safety
    )
    return embeddings

def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a query embedding and a matrix of doc embeddings.
    Assumes embeddings are already normalized (BGE-M3 default).
    """
    return np.dot(doc_embs, query_emb)

from functools import lru_cache

@lru_cache(maxsize=256)
def embed_text_cached(text: str) -> np.ndarray:
    return embed_text(text, show_progress=False)
