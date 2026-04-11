"""
faiss_indexer.py
----------------
FAISS INGESTION PIPELINE — run this ONCE to build the FAISS index.
This is a parallel alternative to embed_articles.py (ChromaDB).
Your existing ChromaDB pipeline is NOT affected.

Steps:
  1. Load all articles (reuses existing load_articles + chunk_articles)
  2. Embed each chunk using Google Gemini gemini-embedding-001
  3. Build a FAISS IndexFlatIP (inner-product / cosine after normalization)
  4. Persist index + metadata to backend/faiss_store/

Usage:
  cd <project root>
  pip install faiss-cpu numpy
  python backend/rag/embedding/faiss_indexer.py
"""

import os
import re
import sys
import time
import json
import pickle
import numpy as np
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import faiss
from google import genai
from google.genai import types as genai_types

from backend.rag.ingestion.load_articles import load_all_articles
from backend.rag.chunking.chunk_articles import chunk_articles

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL     = "models/gemini-embedding-001"
FAISS_STORE_DIR     = PROJECT_ROOT / "backend" / "faiss_store"
INDEX_FILE          = FAISS_STORE_DIR / "algerian_law.index"
METADATA_FILE       = FAISS_STORE_DIR / "algerian_law_metadata.pkl"
BATCH_SIZE          = 20   # same as ChromaDB pipeline — stays within rate limits
INTER_BATCH_SLEEP   = 2    # seconds between batches

# ── Validate API key ──────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. Add it to your .env file:\n"
        "  GEMINI_API_KEY=your_key_here"
    )

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_texts(texts: list) -> np.ndarray:
    """
    Embed a batch of texts with Gemini and return an (N, D) float32 array.
    Vectors are L2-normalized so that inner product == cosine similarity.
    """
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
    # L2-normalize → cosine similarity becomes inner product
    faiss.normalize_L2(vectors)
    return vectors


def _parse_retry_delay(err_str: str, default: int = 60) -> int:
    """Parse retryDelay from a 429 error message."""
    m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err_str)
    return int(m.group(1)) + 5 if m else default


def _load_existing_index():
    """
    Load an existing FAISS index + metadata from disk so the pipeline
    is resumable (skips already-indexed IDs, just like the ChromaDB pipeline).
    Returns (index, metadata_list) or (None, []) if no index exists yet.
    """
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        print(f"[INFO] Loading existing FAISS index from {FAISS_STORE_DIR}")
        index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, "rb") as f:
            metadata_list = pickle.load(f)
        print(f"[INFO] Existing index has {index.ntotal} vectors.")
        return index, metadata_list
    return None, []


def _save_index(index, metadata_list: list):
    """Persist FAISS index and metadata list to disk."""
    FAISS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata_list, f)
    print(f"[INFO] Saved index ({index.ntotal} vectors) → {INDEX_FILE}")
    print(f"[INFO] Saved metadata ({len(metadata_list)} entries) → {METADATA_FILE}")


# ── Main ingestion ─────────────────────────────────────────────────────────────

def run_faiss_ingestion():
    print("=" * 60)
    print("  Algerian Law RAG — FAISS Ingestion Pipeline")
    print("=" * 60)

    # 1. Load + chunk (reuses your existing modules)
    print("\n[1/3] Loading articles from dataset...")
    articles = load_all_articles()

    print("\n[2/3] Chunking articles...")
    chunks = chunk_articles(articles)

    # 2. Resume support — skip already-indexed chunk IDs
    index, metadata_list = _load_existing_index()
    existing_ids = {m["id"] for m in metadata_list}
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    print(f"\n[INFO] {len(existing_ids)} chunks already in FAISS index.")
    print(f"[INFO] {len(new_chunks)} new chunks to index.")

    if not new_chunks:
        print("\n✓ FAISS index is already up-to-date. Nothing to do.")
        return

    # 3. Embed + build / extend FAISS index
    print(f"\n[3/3] Embedding and indexing in batches of {BATCH_SIZE}...")
    total = len(new_chunks)
    indexed = 0

    all_vectors   = []   # accumulate before building index
    all_metadata  = list(metadata_list)  # carry over existing metadata

    for i in range(0, total, BATCH_SIZE):
        batch  = new_chunks[i : i + BATCH_SIZE]
        texts  = [c["text"] for c in batch]

        # Retry loop — mirrors the ChromaDB pipeline's retry logic
        vectors = None
        for attempt in range(6):
            try:
                vectors = embed_texts(texts)
                break
            except Exception as e:
                err_str = str(e)
                wait = _parse_retry_delay(err_str, default=30 * (2 ** attempt))
                if attempt < 5:
                    print(f"[WARN] API error (attempt {attempt+1}/6), waiting {wait}s: {err_str[:100]}")
                    time.sleep(wait)
                else:
                    print(f"[ERROR] Skipping batch after 6 failed attempts.")

        if vectors is None:
            continue

        # Store vectors and rich metadata together
        for j, chunk in enumerate(batch):
            all_vectors.append(vectors[j])
            all_metadata.append({
                "id":       chunk["id"],
                "text":     chunk["text"],
                **chunk["metadata"],   # law_name, article_number, title, etc.
            })

        indexed += len(batch)
        pct = indexed / total * 100
        print(f"  → Embedded {indexed}/{total} ({pct:.1f}%)")

        if i + BATCH_SIZE < total:
            time.sleep(INTER_BATCH_SLEEP)

    if not all_vectors:
        print("[ERROR] No vectors produced. Aborting.")
        return

    # Build FAISS IndexFlatIP (exact search, cosine similarity after L2-norm)
    # If you have >100k docs, swap to IndexIVFFlat for faster approximate search.
    dim = len(all_vectors[0])
    print(f"\n[INFO] Building FAISS IndexFlatIP (dim={dim})...")

    new_matrix = np.array(all_vectors, dtype=np.float32)

    if index is None:
        # Fresh index
        index = faiss.IndexFlatIP(dim)
        # Wrap with IDMap so we can use integer IDs aligned to metadata_list positions
        index = faiss.IndexIDMap(index)

    # FAISS IDs = position in metadata list (existing count + offset)
    start_id = len(metadata_list)  # where new entries begin
    faiss_ids = np.arange(start_id, start_id + len(new_chunks), dtype=np.int64)

    index.add_with_ids(new_matrix, faiss_ids)

    # Persist
    _save_index(index, all_metadata)

    print(f"\n✓ FAISS ingestion complete!")
    print(f"  Total vectors in index : {index.ntotal}")
    print(f"  Total metadata entries : {len(all_metadata)}")
    print(f"  Index location         : {FAISS_STORE_DIR}")


if __name__ == "__main__":
    run_faiss_ingestion()