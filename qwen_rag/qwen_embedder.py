from __future__ import annotations

# ── Make "qwen_rag" importable when run as a plain script ─────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv

# ── env ────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = os.getenv(
    "QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)

EMBED_BATCH_SIZE: int = int(os.getenv("QWEN_EMBED_BATCH_SIZE", "8"))
MAX_SEQ_LEN: int = int(os.getenv("QWEN_EMBED_MAX_SEQ_LEN", "512"))

# PFE Optimization Settings
USE_QUANTIZATION: bool = True  # Set to True to fit in 6GB GPU
FORCE_CPU: bool = False        # Set to True if you still get OOM errors

# Task instruction used when embedding queries (not documents)
_QUERY_INSTRUCTION = (
    "Given a legal question in Arabic, retrieve the most relevant Algerian law "
    "article that answers the question."
)

# ── Lazy singletons ────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None


def _get_model():
    """Load Qwen3-Embedding model with 4-bit quantization to fit 6GB VRAM."""
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    # Device priority
    if FORCE_CPU:
        _device = "cpu"
    elif torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    print(f"[Qwen-Embedder] Loading {EMBEDDING_MODEL_NAME} on {_device} …")

    _tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
    )

    if USE_QUANTIZATION and _device == "cuda":
        print("[Qwen-Embedder] Applying 4-bit quantization to save VRAM...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        _model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # Standard loading (for CPU or high-memory GPUs)
        _model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if _device != "cpu" else torch.float32,
        ).to(_device)

    _model.eval()
    print(f"[Qwen-Embedder] ✓ Model ready (device={_device}, quantized={USE_QUANTIZATION})")
    return _model, _tokenizer, _device


# ── Core encode ────────────────────────────────────────────────────────────────

def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _encode_batch(texts: List[str]) -> np.ndarray:
    model, tokenizer, device = _get_model()

    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(model.device)  # use model.device — bnb may split layers

        with torch.no_grad():
            outputs = model(**encoded)

        embeddings = _last_token_pool(
            outputs.last_hidden_state, encoded["attention_mask"]
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings)


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_documents(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return _encode_batch(texts)


def embed_query(query: str) -> np.ndarray:
    prefixed = f"<instruct>{_QUERY_INSTRUCTION}\n{query}"
    return _encode_batch([prefixed])


def get_embedding_dim() -> int:
    dummy = embed_documents(["test"])
    return dummy.shape[1]


if __name__ == "__main__":
    q_vec = embed_query("ما هي عقوبة السرقة في القانون الجزائري؟")
    d_vec = embed_documents(["يعاقب بالحبس كل من ارتكب جريمة السرقة"])
    sim = float(np.dot(q_vec, d_vec.T))
    print(f"Embedding dim : {q_vec.shape[1]}")
    print(f"Cosine sim    : {sim:.4f}")
