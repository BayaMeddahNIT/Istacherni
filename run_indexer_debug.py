# run_indexer_debug.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import chromadb
from tqdm import tqdm
from bm25_rag.bm25_loader import load_all_articles
from dense_rag.bge_embedder import embed_article, get_model

CHROMA_DIR = Path("dense_rag/chroma_db")
COLLECTION_NAME = "algerian_law"
BATCH_SIZE = 16


def _safe_str(value, max_len: int = None) -> str:
    """Convert any value to a clean string, handling None/list/int."""
    if value is None:
        return ""
    if isinstance(value, list):
        result = " ".join(str(v) for v in value if v is not None)
    else:
        result = str(value)
    if max_len:
        result = result[:max_len]
    return result


def find_problematic_articles(articles: list[dict]) -> None:
    """Scan for articles with None in critical fields."""
    problems = []
    for a in articles:
        issues = []
        for field in ["id", "text_original", "keywords", "title", "law_name"]:
            val = a.get(field)
            if val is None:
                issues.append(f"{field}=None")
            elif isinstance(val, list) and any(v is None for v in val):
                issues.append(f"{field} contains None items")
        if issues:
            problems.append((a.get("id", "?"), issues))

    if problems:
        print(f"\n⚠️  Found {len(problems)} articles with None fields:")
        for aid, issues in problems[:10]:  # show first 10
            print(f"   {aid}: {', '.join(issues)}")
    else:
        print("\n✓ No None field issues found")


# ── Setup ──────────────────────────────────────────────────────────────────────
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection(COLLECTION_NAME)
    print("Deleted old collection")
except Exception:
    print("No old collection to delete")

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"Created collection: {COLLECTION_NAME}")

# ── Load & inspect ─────────────────────────────────────────────────────────────
articles = load_all_articles()
print(f"Loaded {len(articles)} articles")
find_problematic_articles(articles)

# ── Test embedding ─────────────────────────────────────────────────────────────
print("\nTesting embedding on first article...")
try:
    test_embed = embed_article(articles[0])
    print(f"Embedding dim: {len(test_embed)} ✓")
except Exception as e:
    print(f"Embedding FAILED: {e}")
    sys.exit(1)

# ── Pre-warm model so tqdm is clean ───────────────────────────────────────────
get_model()

# ── Index ──────────────────────────────────────────────────────────────────────
success = 0
errors  = 0
error_ids = []

for i in tqdm(range(0, len(articles), BATCH_SIZE), desc="Indexing"):
    batch = articles[i : i + BATCH_SIZE]
    try:
        ids = [_safe_str(a.get("id")) or f"ARTICLE_{i+j}" 
               for j, a in enumerate(batch)]

        embeddings = [embed_article(a) for a in batch]

        metadatas = [
            {
                "id":             _safe_str(a.get("id"))[:100],
                "law_name":       _safe_str(a.get("law_name"))[:200],
                "law_domain":     _safe_str(a.get("law_domain"))[:100],
                "article_number": _safe_str(a.get("article_number"))[:50],
                "title":          _safe_str(a.get("title"))[:200],
                "text_original":  _safe_str(a.get("text_original"))[:500],
                "penalties_summary":
                    _safe_str(a.get("penalties_summary"))[:300],
                "legal_conditions_summary":
                    _safe_str(a.get("legal_conditions_summary"))[:300],
                "keywords":       _safe_str(a.get("keywords"))[:200],
            }
            for a in batch
        ]

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        success += len(batch)

    except Exception as e:
        errors += len(batch)
        error_ids.extend([a.get("id", "?") for a in batch])
        print(f"\n⚠️  Batch {i}–{i+BATCH_SIZE} failed: {e}")
        # Try article by article to isolate the bad one
        for a in batch:
            try:
                aid = _safe_str(a.get("id")) or f"ARTICLE_{i}"
                emb = embed_article(a)
                meta = {
                    "id":             _safe_str(a.get("id"))[:100],
                    "law_name":       _safe_str(a.get("law_name"))[:200],
                    "law_domain":     _safe_str(a.get("law_domain"))[:100],
                    "article_number": _safe_str(a.get("article_number"))[:50],
                    "title":          _safe_str(a.get("title"))[:200],
                    "text_original":  _safe_str(a.get("text_original"))[:500],
                    "penalties_summary":
                        _safe_str(a.get("penalties_summary"))[:300],
                    "legal_conditions_summary":
                        _safe_str(a.get("legal_conditions_summary"))[:300],
                    "keywords":       _safe_str(a.get("keywords"))[:200],
                }
                collection.add(ids=[aid], embeddings=[emb], metadatas=[meta])
                success += 1
                errors  -= 1
            except Exception as e2:
                print(f"   ✗ Skipped article {a.get('id', '?')}: {e2}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"✓ Indexed : {success} articles")
print(f"✗ Failed  : {errors} articles")
print(f"Total in DB: {collection.count()}")

if error_ids:
    print(f"\nProblematic article IDs:")
    for eid in error_ids:
        print(f"  - {eid}")