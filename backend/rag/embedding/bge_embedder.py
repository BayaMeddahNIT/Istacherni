"""
bge_embedder.py
---------------
Shared local embedding utility using BAAI/bge-m3.

BGE-m3 is a multilingual dense retrieval model that supports Arabic,
French, and English — ideal for the Algerian law dataset.

Install dependencies once:
  pip install FlagEmbedding torch

Model is downloaded from HuggingFace on first use (~2 GB) and cached
in ~/.cache/huggingface/. Subsequent runs load from cache.

Embedding dimension : 1024
Max input tokens    : 8192 (covers even long legal articles)
"""

from __future__ import annotations

import numpy as np
from typing import List

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_model = None


def _get_model():
    """Load BAAI/bge-m3 once and reuse for the lifetime of the process."""
    global _model
    if _model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is not installed.\n"
                "Run:  pip install FlagEmbedding torch\n"
                f"Original error: {e}"
            ) from e

        print("[INFO] Loading BAAI/bge-m3 (first run downloads ~2 GB) …")
        _model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,   # halves memory; negligible accuracy loss
        )
        print("[INFO] BGE-m3 model ready.")
    return _model


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_documents(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    """
    Embed a list of document texts for INDEXING.

    Args:
        texts       : List of strings to embed.
        batch_size  : How many texts to encode per forward pass.
                      Lower this if you run out of GPU/CPU memory.
        max_length  : Token limit per text (BGE-m3 supports up to 8192;
                      512 is fast and sufficient for most legal articles).

    Returns:
        np.ndarray of shape (N, 1024), dtype float32, L2-normalised.
    """
    model = _get_model()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vecs = np.array(output["dense_vecs"], dtype=np.float32)
    # Normalise so inner-product == cosine similarity (matches FAISS IndexFlatIP)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
    return vecs / norms


def embed_query(
    query: str,
    max_length: int = 512,
) -> np.ndarray:
    """
    Embed a single query string for RETRIEVAL.

    BGE-m3 uses a different internal instruction for queries vs documents,
    which the model handles internally when `return_dense=True`.

    Returns:
        np.ndarray of shape (1, 1024), dtype float32, L2-normalised.
    """
    vecs = embed_documents([query], batch_size=1, max_length=max_length)
    return vecs  # shape (1, 1024)
