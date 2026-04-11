# dense_rag/bge_indexer.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from tqdm import tqdm
from bm25_rag.bm25_loader import load_all_articles
from dense_rag.bge_embedder import embed_article

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "algerian_law"
BATCH_SIZE = 64  # good for RTX 3060


def _s(value, max_len: int = None) -> str:
    """Null-safe string conversion for any field type."""
    if value is None:
        return ""
    if isinstance(value, list):
        result = " ".join(str(v) for v in value if v is not None)
    else:
        result = str(value)
    return result[:max_len] if max_len else result


def build_vector_index(force: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force:
        print(f"[BGE-Indexer] Collection already exists. Use force=True to rebuild.")
        return client.get_collection(COLLECTION_NAME)

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("[BGE-Indexer] Loading articles…")
    articles = load_all_articles()
    print(f"[BGE-Indexer] Embedding {len(articles)} articles…")

    success = 0
    for i in tqdm(range(0, len(articles), BATCH_SIZE), desc="Indexing"):
        batch = articles[i : i + BATCH_SIZE]
        try:
            ids = [_s(a.get("id")) or f"ARTICLE_{i+j}"
                   for j, a in enumerate(batch)]

            embeddings = [embed_article(a) for a in batch]

            metadatas = [
                {
                    "id":                       _s(a.get("id"), 100),
                    "law_name":                 _s(a.get("law_name"), 200),
                    "law_domain":               _s(a.get("law_domain"), 100),
                    "article_number":           _s(a.get("article_number"), 50),
                    "title":                    _s(a.get("title"), 200),
                    "text_original":            _s(a.get("text_original"), 500),
                    "penalties_summary":        _s(a.get("penalties_summary"), 300),
                    "legal_conditions_summary": _s(a.get("legal_conditions_summary"), 300),
                    "keywords":                 _s(a.get("keywords"), 200),
                }
                for a in batch
            ]

            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            success += len(batch)

        except Exception as e:
            print(f"\n⚠️  Batch {i}–{i+BATCH_SIZE} failed: {e}")
            # Retry one by one to skip only the bad article
            for j, a in enumerate(batch):
                try:
                    aid = _s(a.get("id")) or f"ARTICLE_{i+j}"
                    collection.add(
                        ids=[aid],
                        embeddings=[embed_article(a)],
                        metadatas=[{
                            "id":                       _s(a.get("id"), 100),
                            "law_name":                 _s(a.get("law_name"), 200),
                            "law_domain":               _s(a.get("law_domain"), 100),
                            "article_number":           _s(a.get("article_number"), 50),
                            "title":                    _s(a.get("title"), 200),
                            "text_original":            _s(a.get("text_original"), 500),
                            "penalties_summary":        _s(a.get("penalties_summary"), 300),
                            "legal_conditions_summary": _s(a.get("legal_conditions_summary"), 300),
                            "keywords":                 _s(a.get("keywords"), 200),
                        }]
                    )
                    success += 1
                except Exception as e2:
                    print(f"   ✗ Skipped {a.get('id', '?')}: {e2}")

    print(f"\n[BGE-Indexer] ✓ Indexed {success}/{len(articles)} articles → {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_vector_index(force=True)