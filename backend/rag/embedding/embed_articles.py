"""
embed_articles.py
-----------------
INGESTION PIPELINE — run this ONCE to build the ChromaDB vector store.

Steps:
  1. Load all articles from dataset/raw/**/*.json and *.jsonl
  2. Chunk each article into an embeddable text + metadata
  3. Embed each chunk using Google Gemini gemini-embedding-001
  4. Upsert into a persistent ChromaDB collection (backend/vectorstore/)

Usage:
  cd <project root>
  python backend/rag/embedding/embed_articles.py
"""

import os
import re
import sys
import time
from pathlib import Path

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from google import genai
from google.genai import types as genai_types
import chromadb

from backend.rag.ingestion.load_articles import load_all_articles
from backend.rag.chunking.chunk_articles import chunk_articles

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "algerian_law"
VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore"
BATCH_SIZE = 20          # Smaller batches to reduce rate-limit pressure
INTER_BATCH_SLEEP = 2    # Seconds to sleep between successful batches

# ── Setup ─────────────────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. Create a .env file in the project root with:\n"
        "  GEMINI_API_KEY=your_key_here"
    )

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Persistent ChromaDB client
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def embed_texts(texts: list) -> list:
    """
    Embed a batch of texts using Gemini gemini-embedding-001.
    Returns a list of embedding vectors (one per text).
    """
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in response.embeddings]


def _parse_retry_delay(err_str: str, default: int = 60) -> int:
    """Parse the retryDelay seconds from a 429 error message."""
    m = re.search(r"retryDelay['\"]?\s*[:'\"]+\s*['\"]?(\d+)s", err_str)
    return int(m.group(1)) + 5 if m else default


def run_ingestion():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("  Algerian Law RAG — Ingestion Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/3] Loading articles from dataset...")
    articles = load_all_articles()

    # 2. Chunk
    print("\n[2/3] Chunking articles...")
    chunks = chunk_articles(articles)

    # Check which IDs are already in the collection (for resumable ingestion)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    print(f"\n[INFO] {len(existing_ids)} articles already in vector store.")
    print(f"[INFO] {len(new_chunks)} new articles to index.")

    if not new_chunks:
        print("\n✓ Vector store is already up-to-date. Nothing to do.")
        return

    # 3. Embed + upsert in batches
    print(f"\n[3/3] Embedding and upserting in batches of {BATCH_SIZE}...")
    total = len(new_chunks)
    indexed = 0

    for i in range(0, total, BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Retry loop — honors the retryDelay hint from 429 responses
        embeddings = None
        for attempt in range(6):
            try:
                embeddings = embed_texts(texts)
                break
            except Exception as e:
                err_str = str(e)
                wait = _parse_retry_delay(err_str, default=30 * (2 ** attempt))
                if attempt < 5:
                    print(f"\n[WARN] API error (attempt {attempt+1}/6), waiting {wait}s: {err_str[:100]}")
                    time.sleep(wait)
                else:
                    print(f"\n[ERROR] Skipping batch after 6 failed attempts.")

        if embeddings is None:
            continue

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        indexed += len(batch)
        pct = indexed / total * 100
        print(f"  → Indexed {indexed}/{total} ({pct:.1f}%)")

        # Small pause between batches to avoid rate limits
        if i + BATCH_SIZE < total:
            time.sleep(INTER_BATCH_SLEEP)

    print(f"\n\n✓ Ingestion complete! {indexed} articles indexed.")
    print(f"  Collection '{COLLECTION_NAME}' now has {collection.count()} total documents.")
    print(f"  Vector store: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    run_ingestion()
