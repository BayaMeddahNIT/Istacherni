from __future__ import annotations
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

# ── Paths ───────────────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "algerian_law_camelbert"

# ── Document text builder ───────────────────────────────────────────────────────

def build_document_text(article: dict) -> str:
    """Combine several article fields into a single string for embedding."""
    parts = [
        article.get("title", ""),
        article.get("title", ""),           # ×2 weight
        article.get("text_original", ""),
        article.get("summary", ""),
        article.get("legal_conditions_summary", ""),
        article.get("penalties_summary", ""),
        " ".join(article.get("keywords", [])),
        " ".join(article.get("keywords", [])),  # ×2 weight
    ]
    return " ".join(p for p in parts if p).strip()

# ── Index builder ────────────────────────────────────────────────────────────────

def build_index(force: bool = False):
    """
    Build the ChromaDB collection and persist it to disk.
    """
    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    # Handle force rebuild
    if force:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"[Chroma-Indexer] Deleted existing collection: {COLLECTION_NAME}")
        except ValueError:
            pass

    # Create or get collection
    # We use 'cosine' space to match your previous FAISS IndexFlatIP logic
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        metadata={"hnsw:space": "cosine"}
    )

    # Check if already indexed
    if not force and collection.count() > 0:
        print(f"[Chroma-Indexer] Index already exists with {collection.count()} docs.")
        return collection

    # ── Load articles ──────────────────────────────────────────────────────────
    try:
        from bm25_rag.bm25_loader import load_all_articles
    except ImportError:
        print("[Chroma-Indexer] ERROR: Could not import bm25_rag.bm25_loader.", file=sys.stderr)
        raise

    print("[Chroma-Indexer] Loading articles …")
    articles = load_all_articles()

    texts_valid = []
    metadatas_valid = []
    ids_valid = []

    for i, a in enumerate(articles):
        text = build_document_text(a)
        if text.strip():
            texts_valid.append(text)
            # Chroma metadata must be flat (strings, ints, floats, bools)
            # We store the article ID and title for easy retrieval
            metadatas_valid.append({
                "law_name": a.get("law_name", ""),
                "law_domain": a.get("law_domain", ""),
                "article_number": a.get("article_number", ""),
                "title": a.get("title", ""),
                "penalties_summary": a.get("penalties_summary", ""),
                "legal_conditions_summary": a.get("legal_conditions_summary", "")
                # Keywords should be converted to a string if they are a list, 
                # as Chroma metadata doesn't support lists.
            })
            ids_valid.append(f"art_{i}")

    print(f"[Chroma-Indexer] Embedding and Indexing {len(texts_valid)} articles …")

    # Import your custom embedder
    from camelbert_rag.camelbert_embedder import embed_texts
    embeddings = embed_texts(texts_valid, batch_size=32, normalize=True)

    # ── Add to Chroma ──────────────────────────────────────────────────────────
    # Chroma stores the text, the vector, and the metadata together
    collection.add(
        embeddings=embeddings.tolist(), # Chroma expects a list of lists
        documents=texts_valid,
        metadatas=metadatas_valid,
        ids=ids_valid
    )

    print(f"[Chroma-Indexer] ✓ Chroma collection saved to {CHROMA_PATH}")
    print(f"[Chroma-Indexer] Total docs indexed: {collection.count()}\n")

    return collection

# ── CLI entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the CAMeLBERT Chroma index.")
    parser.add_argument("--force", action="store_true", help="Rebuild the index.")
    args = parser.parse_args()

    coll = build_index(force=args.force)
    print(f"Collection count: {coll.count()}")