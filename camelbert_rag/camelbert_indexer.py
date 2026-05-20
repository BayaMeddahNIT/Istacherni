from __future__ import annotations
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings
import re

# ── Paths ───────────────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "algerian_law_camelbert"

# ── Document text builder ───────────────────────────────────────────────────────

def normalize_for_camelbert(text: str) -> str:
    if not text: return ""
    # Eastern Arabic → Western Arabic numerals
    eastern = '٠١٢٣٤٥٦٧٨٩'
    western = '0123456789'
    trans = str.maketrans(eastern, western)
    text = text.translate(trans)
    
    # Normalize article reference format
    text = re.sub(r'المادة\s+(\d+)', r'المادة \1', text)
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    
    # Normalize Alef, Ya, Ta Marbuta
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ى', 'ي')
    text = text.replace('ة', 'ه')
    
    return text

def build_document_text(article: dict) -> str:
    """
    Build a clean, coherent passage for embedding.

    Uses only title + original legal text — the most semantically stable
    representation for a dense retrieval model.
    """
    title = article.get("title", "").strip()
    text  = article.get("text_original", "").strip()
    
    law_name = article.get("law_name", "قانون جزائري")
    article_number = str(article.get("article_number", "") or "").strip()
    
    prefix = f"[{law_name} المادة {article_number}] " if article_number else ""
    
    combined = f"{title}. {text}" if title and text else (title or text)
    
    return normalize_for_camelbert(prefix + combined)


def build_bm25_text(article: dict) -> str:
    """
    Build a keyword-enriched text string for BM25 / full-text search.
    This intentionally includes keywords and summaries that would pollute
    a dense embedding but are useful for lexical retrieval.
    """
    parts = [
        article.get("title", ""),
        article.get("text_original", ""),
        article.get("summary", ""),
        article.get("legal_conditions_summary", ""),
        article.get("penalties_summary", ""),
        " ".join(article.get("keywords", [])),
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
        if not text.strip():
            continue
        texts_valid.append(text)
        # Ensure article_number is always a non-empty string.
        # An empty string here causes sources to be written as "law_name - المادة " (no number).
        art_num = str(a.get("article_number", "") or "").strip()
        metadatas_valid.append({
            "law_name":                    str(a.get("law_name", "") or "قانون جزائري"),
            "law_domain":                  str(a.get("law_domain", "") or ""),
            "article_number":              art_num,
            "title":                       str(a.get("title", "") or ""),
            "penalties_summary":           str(a.get("penalties_summary", "") or ""),
            "legal_conditions_summary":    str(a.get("legal_conditions_summary", "") or ""),
            "keywords_text":               " ".join(a.get("keywords", []) or []),
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