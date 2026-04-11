"""
camelbert_embedder.py
---------------------
Embedding module for the CameLBERT dense-vector RAG.

Uses CAMeL-Lab/camelbert-mix-ner (or any CAMeLBERT variant) to produce
sentence-level embeddings via mean-pooling over the last hidden states.

Why CAMeLBERT-mix?
  - Trained on a massive, diverse Arabic corpus covering Modern Standard Arabic
    (MSA), dialects, and multiple domains.
  - Legal Arabic is formal MSA → CAMeLBERT-mix generalises better than
    dialect-specific models.
  - Outperforms AraBERT on most Arabic NLP benchmarks for formal text.

Model used: CAMeL-Lab/camelbert-mix
  • ~110 M parameters (BERT-base sized)
  • Max sequence length: 512 tokens
  • Hidden size: 768 → embedding dim = 768

Completely standalone — no dependency on bm25_rag or the Gemini-based code.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ── Model identifier ────────────────────────────────────────────────────────────
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

# ── Lazy singletons ─────────────────────────────────────────────────────────────
_tokenizer = None
_model = None
_device = None


def _get_model():
    """Load tokeniser + model once; reuse on subsequent calls."""
    global _tokenizer, _model, _device

    if _model is None:
        print(f"[CameLBERT-Embedder] Loading model '{MODEL_NAME}' …")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = _model.to(_device)
        _model.eval()
        print(f"[CameLBERT-Embedder] Model loaded on {_device}.")

    return _tokenizer, _model, _device


# ── Core embedding function ─────────────────────────────────────────────────────

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings, ignoring padding tokens.

    Args:
        last_hidden_state: (batch, seq_len, hidden_size)
        attention_mask:    (batch, seq_len)  — 1 for real tokens, 0 for padding

    Returns:
        (batch, hidden_size) tensor of sentence embeddings.
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()          # (B, S, 1)
    summed = (last_hidden_state * mask_expanded).sum(dim=1)       # (B, H)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)             # (B, 1)
    return summed / counts                                         # (B, H)


def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute CAMeLBERT embeddings for a list of Arabic strings.

    Args:
        texts:      List of Arabic strings to embed.
        batch_size: Number of texts processed per forward pass.
        max_length: Maximum tokenisation length (≤ 512 for BERT-based models).
        normalize:  If True, L2-normalise each vector (makes cosine similarity
                    equivalent to dot-product, and is recommended for retrieval).

    Returns:
        np.ndarray of shape (len(texts), 768), dtype float32.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    tokenizer, model, device = _get_model()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        pooled = _mean_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )  # (batch, 768)

        embeddings = pooled.cpu().numpy().astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)  # (N, 768)


def embed_query(query: str, max_length: int = 512, normalize: bool = True) -> np.ndarray:
    """
    Convenience wrapper: embed a single query string.

    Returns:
        np.ndarray of shape (768,), dtype float32.
    """
    return embed_texts([query], max_length=max_length, normalize=normalize)[0]


# ── Smoke test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "يعاقب بالسجن المؤقت من عشر سنوات إلى عشرين سنة.",
    ]
    vecs = embed_texts(sample)
    print(f"Shape : {vecs.shape}")          # (2, 768)
    print(f"Norm  : {np.linalg.norm(vecs[0]):.6f}")   # should be ≈ 1.0 when normalised
    sim = float(np.dot(vecs[0], vecs[1]))
    print(f"Cosine similarity (q ↔ article) : {sim:.4f}")