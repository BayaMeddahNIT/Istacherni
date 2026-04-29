"""
qwen_embed_articles.py
----------------------
INGESTION PIPELINE — run this ONCE to build the ChromaDB vector store
using a LOCAL Qwen3-Embedding-0.6B model (no API key required).

Steps:
  1. Load all articles from dataset/raw/**/*.json
  2. Chunk each article into an embeddable text + metadata
  3. Embed each chunk using Qwen/Qwen3-Embedding-0.6B (local, via sentence-transformers)
  4. Upsert into a persistent ChromaDB collection (backend/vectorstore_qwen/)

Usage:
  cd <project root>
  pip install sentence-transformers chromadb

  python backend/rag/embedding/qwen_embed_articles.py

Notes:
  - First run downloads ~500 MB from HuggingFace Hub (cached afterwards).
  - Uses cosine similarity space in ChromaDB (same as the Gemini pipeline).
  - A separate collection name ("algerian_law_qwen") is used to avoid
    collisions with any existing Gemini-based vector store.
  - Supports resumable ingestion: already-indexed IDs are skipped.
"""

import os
import sys
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import chromadb
from sentence_transformers import SentenceTransformer

from backend.rag.ingestion.load_articles import load_all_articles
from backend.rag.chunking.chunk_articles import chunk_articles

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "algerian_law_qwen"
VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore_qwen"
BATCH_SIZE      = 32        # Sentence-transformers handles batching internally,
                             # but we chunk upserts to keep memory predictable.

# Qwen3-Embedding retrieval task prompt (prepended to every passage at index time)
# Using the instruction format recommended by the Qwen team for asymmetric retrieval.
DOCUMENT_PROMPT = (
    "Instruct: Given a legal article, represent it for retrieval.\nPassage: "
)

# ── Model — loaded once at module level ───────────────────────────────────────
print(f"[INFO] Loading embedding model: {MODEL_NAME}")
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {_device}")

embedding_model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    device=_device,
)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts using Qwen3-Embedding-0.6B.

    The DOCUMENT_PROMPT is prepended to every text so the model encodes them
    in 'passage' mode, matching the asymmetric retrieval setup in qwen_retriever.py.

    Returns a list of normalized float vectors (one per text).
    """
    prompted = [DOCUMENT_PROMPT + t for t in texts]
    vectors = embedding_model.encode(
        prompted,
        normalize_embeddings=True,   # cosine sim ≡ dot product after L2-norm
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
    )
    return vectors.tolist()


def run_ingestion() -> None:
    """Main ingestion pipeline."""
    print("=" * 60)
    print("  Algerian Law RAG — Local Qwen3 Ingestion Pipeline")
    print("=" * 60)

    # 1. Load ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading articles from dataset...")
    articles = load_all_articles()

    # 2. Chunk ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Chunking articles...")
    chunks = chunk_articles(articles)

    # Resumable: skip IDs that are already indexed
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["id"] not in existing_ids]

    print(f"\n[INFO] {len(existing_ids)} articles already in vector store.")
    print(f"[INFO] {len(new_chunks)} new articles to index.")

    if not new_chunks:
        print("\n✓ Vector store is already up-to-date. Nothing to do.")
        return

    # 3. Embed + upsert in batches ─────────────────────────────────────────────
    print(f"\n[3/3] Embedding and upserting in batches of {BATCH_SIZE}...")
    total   = len(new_chunks)
    indexed = 0

    for i in range(0, total, BATCH_SIZE):
        batch     = new_chunks[i : i + BATCH_SIZE]
        texts     = [c["text"]     for c in batch]
        ids       = [c["id"]       for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embed_documents(texts)

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas,
        )

        indexed += len(batch)
        pct = indexed / total * 100
        print(f"  → Indexed {indexed}/{total} ({pct:.1f}%)")

    print(f"\n\n✓ Ingestion complete! {indexed} articles indexed.")
    print(f"  Collection '{COLLECTION_NAME}' now has {collection.count()} total documents.")
    print(f"  Vector store: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    run_ingestion()