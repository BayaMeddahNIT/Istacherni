"""
camelbert_embed_articles.py
----------------------------
INGESTION PIPELINE (local CamELBERT variant) — run this ONCE to build a
separate ChromaDB vector store using a locally-downloaded Arabic BERT model.

Model used: CAMeL-Lab/bert-base-arabic-camelbert-mix
  → Best general-purpose CamELBERT variant (trained on MSA + dialectal Arabic).
  → For purely Modern Standard Arabic (MSA) you can swap to:
       CAMeL-Lab/bert-base-arabic-camelbert-msa
  → Embeddings are produced via mean-pooling over the last hidden state.

Steps:
  1. Load all articles from dataset/raw/**/*.json
  2. Chunk each article into embeddable text + metadata
  3. Embed each chunk with CamELBERT (mean-pool, L2-normalised)
  4. Upsert into a persistent ChromaDB collection  (backend/vectorstore_camelbert/)

Usage:
  cd <project root>
  pip install transformers torch chromadb
  python backend/rag/embedding/camelbert_embed_articles.py
"""

import os
import sys
import time
from pathlib import Path

print("Initializing CamELBERT embedding pipeline... Please wait while heavy libraries load.", flush=True)

import torch
import chromadb
from transformers import AutoTokenizer, AutoModel

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.ingestion.load_articles import load_all_articles
from backend.rag.chunking.chunk_articles import chunk_articles

# ── Configuration ──────────────────────────────────────────────────────────────
CAMELBERT_MODEL   = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
COLLECTION_NAME   = "algerian_law_camelbert"
VECTORSTORE_DIR   = PROJECT_ROOT / "backend" / "vectorstore_camelbert"
BATCH_SIZE        = 32          # Safe for ~8 GB VRAM; lower to 16 if OOM on CPU
MAX_SEQ_LEN       = 512         # CamELBERT's hard limit
INTER_BATCH_SLEEP = 0           # No rate-limit needed for local inference

# ── Device selection ───────────────────────────────────────────────────────────
DEVICE = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"[INFO] Using device: {DEVICE}")


# ── Model singleton ────────────────────────────────────────────────────────────
_tokenizer = None
_model     = None


def _get_model():
    """Lazy-load the CamELBERT tokenizer and model (downloaded once, cached by HF)."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"[INFO] Loading CamELBERT model: {CAMELBERT_MODEL}  (first run downloads ~500 MB)")
        _tokenizer = AutoTokenizer.from_pretrained(CAMELBERT_MODEL)
        _model     = AutoModel.from_pretrained(CAMELBERT_MODEL).to(DEVICE)
        _model.eval()
    return _tokenizer, _model


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings, ignoring padding tokens.
    Shape: (batch_size, hidden_dim)
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask       = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Produce L2-normalised mean-pool embeddings for a batch of texts.

    Returns a list of plain Python float lists (one per text), compatible
    with ChromaDB's `embeddings` argument.
    """
    tokenizer, model = _get_model()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)

    pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

    # L2 normalise so cosine similarity == dot product (better for ChromaDB)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled.cpu().tolist()


# ── ChromaDB setup ─────────────────────────────────────────────────────────────
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection    = chroma_client.get_or_create_collection(
    name     = COLLECTION_NAME,
    metadata = {"hnsw:space": "cosine"},
)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_ingestion():
    print("=" * 60)
    print("  Algerian Law RAG — CamELBERT Ingestion Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/3] Loading articles from dataset...")
    articles = load_all_articles()

    # 2. Chunk
    print("\n[2/3] Chunking articles...")
    chunks = chunk_articles(articles)

    # Resume support: skip already-indexed IDs
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["id"] not in existing_ids]

    print(f"\n[INFO] {len(existing_ids)} articles already in vector store.")
    print(f"[INFO] {len(new_chunks)} new articles to index.")

    if not new_chunks:
        print("\n✓ Vector store is already up-to-date. Nothing to do.")
        return

    # 3. Embed + upsert
    print(f"\n[3/3] Embedding and upserting in batches of {BATCH_SIZE}...")
    total   = len(new_chunks)
    indexed = 0

    for i in range(0, total, BATCH_SIZE):
        batch     = new_chunks[i : i + BATCH_SIZE]
        texts     = [c["text"]     for c in batch]
        ids       = [c["id"]       for c in batch]
        metadatas = [c["metadata"] for c in batch]

        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            print(f"\n[ERROR] Embedding failed for batch {i}–{i+BATCH_SIZE}: {e}")
            print("        Skipping this batch and continuing.")
            continue

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas,
        )

        indexed += len(batch)
        pct      = indexed / total * 100
        print(f"  → Indexed {indexed}/{total} ({pct:.1f}%)")

        if INTER_BATCH_SLEEP:
            time.sleep(INTER_BATCH_SLEEP)

    print(f"\n\n✓ Ingestion complete! {indexed} articles indexed.")
    print(f"  Collection '{COLLECTION_NAME}' now has {collection.count()} total documents.")
    print(f"  Vector store: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    run_ingestion()