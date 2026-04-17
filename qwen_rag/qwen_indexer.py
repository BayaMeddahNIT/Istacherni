"""
qwen_indexer.py
---------------
Builds (and persists) a ChromaDB vector store from the Algerian law dataset
using Qwen3-Embedding-8B embeddings.

Run ONCE before querying:
    python qwen_rag/qwen_indexer.py

Output:
    qwen_rag/chroma_db/   <- persistent ChromaDB directory

The collection is named  COLLECTION_NAME  (see constant below).
Re-running with  force=True  drops and rebuilds the collection.
"""

from __future__ import annotations

# ── Make "qwen_rag" importable when run as a plain script ─────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from typing import List

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DIR:      Path = Path(__file__).parent / "chroma_db"
COLLECTION_NAME: str  = "algerian_law_qwen3"
UPSERT_BATCH:    int  = 64


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_doc_text(article: dict) -> str:
    """Concatenate rich fields for embedding. Keywords repeated ×2 for signal."""
    kw = " ".join(article.get("keywords", []))
    parts = [
        article.get("title", ""),
        article.get("text_original", ""),
        article.get("summary", ""),
        article.get("legal_conditions_summary", ""),
        article.get("penalties_summary", ""),
        kw,
        kw,  # × 2
    ]
    return " ".join(p for p in parts if p).strip()


def _get_collection(client, force: bool = False):
    """Return (or create) the ChromaDB collection."""
    existing = [c.name for c in client.list_collections()]

    if force and COLLECTION_NAME in existing:
        print(f"[Qwen-Indexer] Dropping existing collection '{COLLECTION_NAME}' …")
        client.delete_collection(COLLECTION_NAME)
        existing = []

    if COLLECTION_NAME not in existing:
        print(f"[Qwen-Indexer] Creating collection '{COLLECTION_NAME}' …")
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(COLLECTION_NAME)


# ── Main build ─────────────────────────────────────────────────────────────────

def build_index(force: bool = False):
    """
    Build the ChromaDB index.

    Args:
        force: If True, drop existing collection and rebuild from scratch.

    Returns:
        The ChromaDB collection object.
    """
    import chromadb
    from qwen_rag.qwen_loader import load_all_articles
    from qwen_rag.qwen_embedder import embed_documents

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = _get_collection(client, force=force)

    if not force and collection.count() > 0:
        print(
            f"[Qwen-Indexer] Collection already has {collection.count()} docs. "
            "Pass force=True to rebuild."
        )
        return collection

    print("[Qwen-Indexer] Loading articles …")
    articles = load_all_articles()

    doc_texts: List[str] = [_build_doc_text(a) for a in articles]

    valid = [(t, a) for t, a in zip(doc_texts, articles) if t]
    if not valid:
        raise RuntimeError("No valid documents found in dataset.")
    doc_texts_valid, articles_valid = zip(*valid)

    total = len(articles_valid)
    print(f"[Qwen-Indexer] Embedding {total} articles …")

    for batch_start in range(0, total, UPSERT_BATCH):
        batch_end   = min(batch_start + UPSERT_BATCH, total)
        batch_texts = list(doc_texts_valid[batch_start:batch_end])
        batch_arts  = list(articles_valid[batch_start:batch_end])

        embeddings = embed_documents(batch_texts).tolist()

        ids       = [a["id"] for a in batch_arts]
        documents = [a.get("text_original", "") for a in batch_arts]
        metadatas = [
            {
                "law_name":                 a.get("law_name", ""),
                "law_domain":               a.get("law_domain", ""),
                "article_number":           str(a.get("article_number", "")),
                "title":                    a.get("title", ""),
                "penalties_summary":        a.get("penalties_summary", ""),
                "legal_conditions_summary": a.get("legal_conditions_summary", ""),
                "keywords":                 json.dumps(
                    a.get("keywords", []), ensure_ascii=False
                ),
            }
            for a in batch_arts
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        pct = batch_end / total * 100
        print(f"  … {batch_end}/{total}  ({pct:.1f}%)", end="\r", flush=True)

    print(f"\n[Qwen-Indexer] ✓ Indexed {collection.count()} articles → {CHROMA_DIR}")
    return collection


def load_collection():
    """Load the existing ChromaDB collection (must call build_index first)."""
    import chromadb

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_DIR}\n"
            "Run:  python qwen_rag/qwen_indexer.py"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col    = client.get_collection(COLLECTION_NAME)
    print(f"[Qwen-Indexer] Loaded collection '{COLLECTION_NAME}' ({col.count()} docs)")
    return col


if __name__ == "__main__":
    build_index(force=True)
