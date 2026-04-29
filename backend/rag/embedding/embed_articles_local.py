"""
embed_articles_local.py
-----------------------
INGESTION PIPELINE — local embedding variant.
Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 instead
of the Gemini API, so no API key is required for embedding.

Steps:
  1. Load all articles from dataset/raw/**/*.json
  2. Chunk each article into an embeddable text + metadata
  3. Embed each chunk locally with paraphrase-multilingual-MiniLM-L12-v2
  4. Upsert into a persistent ChromaDB collection (backend/vectorstore_local/)

Usage:
  cd <project root>
  pip install sentence-transformers chromadb
  python backend/rag/embedding/embed_articles_local.py

NOTE: On first run the model weights (~120 MB) are downloaded and cached
      automatically by HuggingFace. Subsequent runs use the local cache.

⚠  This pipeline writes to a SEPARATE collection directory
   (backend/vectorstore_local/) so it never touches the Gemini-based store.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import chromadb
from sentence_transformers import SentenceTransformer

from backend.rag.ingestion.load_articles import load_all_articles
from backend.rag.chunking.chunk_articles import chunk_articles

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME      = "algerian_law_local"
VECTORSTORE_DIR      = PROJECT_ROOT / "backend" / "vectorstore_local"
BATCH_SIZE           = 64   # Local inference is fast; large batches are fine

# ── Setup ─────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME}")
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def embed_texts(texts: list) -> list:
    """
    Embed a batch of texts using the local sentence-transformers model.
    Returns a list of embedding vectors (one per text).
    """
    embeddings = _embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # unit-normalise for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def run_ingestion():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("  Algerian Law RAG — Ingestion Pipeline (Local Embeddings)")
    print("=" * 60)

    # 1. Load
    print("\n[1/3] Loading articles from dataset...")
    articles = load_all_articles()

    # 2. Chunk
    print("\n[2/3] Chunking articles...")
    chunks = chunk_articles(articles)

    # Resume-safe: skip IDs already present in the collection
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["id"] not in existing_ids]

    print(f"\n[INFO] {len(existing_ids)} articles already in vector store.")
    print(f"[INFO] {len(new_chunks)} new articles to index.")

    if not new_chunks:
        print("\n✓ Vector store is already up-to-date. Nothing to do.")
        return

    # 3. Embed + upsert in batches
    print(f"\n[3/3] Embedding and upserting in batches of {BATCH_SIZE}...")
    total   = len(new_chunks)
    indexed = 0

    for i in range(0, total, BATCH_SIZE):
        batch     = new_chunks[i : i + BATCH_SIZE]
        texts     = [c["text"]     for c in batch]
        ids       = [c["id"]       for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embed_texts(texts)

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas,
        )

        indexed += len(batch)
        pct      = indexed / total * 100
        print(f"  → Indexed {indexed}/{total} ({pct:.1f}%)")

    print(f"\n\n✓ Ingestion complete! {indexed} articles indexed.")
    print(f"  Collection '{COLLECTION_NAME}' now has {collection.count()} total documents.")
    print(f"  Vector store: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    run_ingestion()