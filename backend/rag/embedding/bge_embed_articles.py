"""
bge_embed_articles.py
---------------------
ChromaDB INGESTION PIPELINE using local BAAI/bge-m3 embeddings.
This is a drop-in replacement for embed_articles.py (Gemini) —
the ChromaDB collection schema is identical so both pipelines
write to the same store (different collection names to avoid
conflicts during migration).

Steps:
  1. Load all articles from dataset/raw/**/*.json
  2. Chunk each article into embeddable text + metadata
  3. Embed with BAAI/bge-m3 (local, no API key needed)
  4. Upsert into persistent ChromaDB collection: algerian_law_bge

Usage:
  pip install FlagEmbedding torch chromadb
  cd <project root>
  python backend/rag/embedding/bge_embed_articles.py
"""

import sys
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

from backend.rag.ingestion.load_articles import load_all_articles
from backend.rag.chunking.chunk_articles import chunk_articles
from backend.rag.embedding.bge_embedder import embed_documents

# ── Configuration ─────────────────────────────────────────────────────────────
COLLECTION_NAME  = "algerian_law_bge"   # separate from the Gemini collection
VECTORSTORE_DIR  = PROJECT_ROOT / "backend" / "vectorstore"
BATCH_SIZE       = 64    # BGE-m3 on CPU: 32–64 is a good balance
                          # If you have a GPU you can safely raise this to 128+

# ── ChromaDB client ───────────────────────────────────────────────────────────
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def run_ingestion():
    print("=" * 60)
    print("  Algerian Law RAG — BGE-m3 ChromaDB Ingestion")
    print("=" * 60)

    # 1. Load
    print("\n[1/3] Loading articles …")
    articles = load_all_articles()

    # 2. Chunk
    print("\n[2/3] Chunking articles …")
    chunks = chunk_articles(articles)

    # Resume support — skip already-indexed IDs
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["id"] not in existing_ids]

    print(f"\n[INFO] Already in vector store : {len(existing_ids)}")
    print(f"[INFO] New chunks to index     : {len(new_chunks)}")

    if not new_chunks:
        print("\n✓ Vector store is already up-to-date. Nothing to do.")
        return

    # 3. Embed + upsert
    print(f"\n[3/3] Embedding with BGE-m3 in batches of {BATCH_SIZE} …")
    total   = len(new_chunks)
    indexed = 0

    for i in range(0, total, BATCH_SIZE):
        batch     = new_chunks[i : i + BATCH_SIZE]
        texts     = [c["text"]     for c in batch]
        ids       = [c["id"]       for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # embed_documents returns float32 ndarray (N, 1024), L2-normalised
        embeddings = embed_documents(texts).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        indexed += len(batch)
        pct = indexed / total * 100
        print(f"  → Indexed {indexed}/{total}  ({pct:.1f} %)")

    print(f"\n✓ Ingestion complete!  {indexed} articles indexed.")
    print(f"  Collection '{COLLECTION_NAME}' now has "
          f"{collection.count()} total documents.")
    print(f"  Vector store: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    run_ingestion()
