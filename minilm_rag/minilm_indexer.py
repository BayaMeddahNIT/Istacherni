"""
minilm_indexer.py
-----------------
Builds and saves TWO indexes over the Algerian law dataset:
  1. BM25Okapi index       → index/bm25_index.pkl  + index/bm25_corpus.pkl
  2. MiniLM dense index    → index/minilm_embeddings.npy + index/minilm_corpus.pkl

Run ONCE before querying:
  python -m minilm_rag.minilm_indexer

The embedding model used is:
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (~120 MB, multilingual, supports Arabic + French)

This module is completely standalone — it does NOT touch bm25_rag/.
It reuses bm25_rag.bm25_loader only to share the same article loading logic.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Reuse the shared loader (does NOT modify bm25_rag in any way)
from bm25_rag.bm25_loader import load_all_articles

# ── Paths ──────────────────────────────────────────────────────────────────────
INDEX_DIR       = Path(__file__).parent / "index"
BM25_FILE       = INDEX_DIR / "bm25_index.pkl"
BM25_CORPUS     = INDEX_DIR / "bm25_corpus.pkl"
EMBED_FILE      = INDEX_DIR / "minilm_embeddings.npy"
EMBED_CORPUS    = INDEX_DIR / "minilm_corpus.pkl"

# ── Model ──────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 64          # safe for most machines; raise if you have GPU


# ── Arabic tokeniser (identical logic to bm25_rag, kept local for independence) ──
def tokenize_arabic(text: str) -> list[str]:
    """
    Lightweight Arabic tokeniser for BM25:
      1. Strip diacritics and tatweel
      2. Normalise letter variants  (أإآ→ا, ة→ه, ى→ي)
      3. Split on non-word characters
      4. Drop tokens shorter than 2 chars
    """
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ة",      "ه", text)
    text = re.sub(r"ى",      "ي", text)
    tokens = re.split(r"[^\w\u0600-\u06FF]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ── Document text builders ─────────────────────────────────────────────────────
def build_bm25_text(article: dict) -> str:
    """
    Rich BM25 document: title + full text + summary + conditions + penalties.
    Keywords repeated twice to boost their BM25 weight.
    """
    kw = " ".join(article.get("keywords", []))
    parts = [
        article.get("title", ""),
        article.get("text_original", ""),
        article.get("summary", ""),
        article.get("legal_conditions_summary", ""),
        article.get("penalties_summary", ""),
        kw,   # first pass
        kw,   # second pass  (×2 weight)
    ]
    return " ".join(p for p in parts if p)


def build_embed_text(article: dict) -> str:
    """
    Sentence to embed: concise but rich enough for semantic search.
    Format: "<title> | <text_original> | <summary>"
    Keeps the string short so MiniLM (max 128 tokens) doesn't truncate too much.
    """
    parts = [
        article.get("title", ""),
        article.get("text_original", ""),
        article.get("summary", ""),
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


# ── Main builder ───────────────────────────────────────────────────────────────
def build_index(force: bool = False) -> dict:
    """
    Build (or load) both indexes.

    Args:
        force: If True, rebuild even when cached files exist.

    Returns:
        {
          "bm25":       BM25Okapi,
          "bm25_corpus": list[dict],
          "embeddings":  np.ndarray  shape (N, 384),
          "embed_corpus": list[dict],
        }
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_cached = (
        BM25_FILE.exists() and BM25_CORPUS.exists()
        and EMBED_FILE.exists() and EMBED_CORPUS.exists()
    )

    if not force and all_cached:
        print("[MiniLM-Indexer] All indexes found on disk — loading…")
        return _load_indexes()

    # ── 1. Load articles ───────────────────────────────────────────────────────
    print("[MiniLM-Indexer] Loading articles from dataset…")
    articles = load_all_articles()

    # ── 2. BM25 ───────────────────────────────────────────────────────────────
    print("[MiniLM-Indexer] Tokenising for BM25…")
    tokenized = [tokenize_arabic(build_bm25_text(a)) for a in articles]

    valid_pairs = [(tok, art) for tok, art in zip(tokenized, articles) if tok]
    if not valid_pairs:
        raise ValueError("No valid articles found — check your dataset path.")

    tokenized_valid, articles_valid = zip(*valid_pairs)
    articles_valid = list(articles_valid)

    print(f"[MiniLM-Indexer] Building BM25Okapi over {len(articles_valid)} docs…")
    bm25 = BM25Okapi(list(tokenized_valid))

    with open(BM25_FILE,   "wb") as f: pickle.dump(bm25,           f)
    with open(BM25_CORPUS, "wb") as f: pickle.dump(articles_valid, f)
    print(f"[MiniLM-Indexer] ✓ BM25 index saved → {BM25_FILE}")

    # ── 3. Dense embeddings ───────────────────────────────────────────────────
    print(f"\n[MiniLM-Indexer] Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    embed_texts = [build_embed_text(a) for a in articles_valid]

    print(f"[MiniLM-Indexer] Encoding {len(embed_texts)} articles "
          f"(batch_size={EMBED_BATCH_SIZE})…")
    embeddings = model.encode(
        embed_texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,    # unit-norm → cosine sim = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(EMBED_FILE, embeddings)
    with open(EMBED_CORPUS, "wb") as f: pickle.dump(articles_valid, f)
    print(f"[MiniLM-Indexer] ✓ Embeddings saved → {EMBED_FILE}  "
          f"shape={embeddings.shape}")

    return {
        "bm25":         bm25,
        "bm25_corpus":  articles_valid,
        "embeddings":   embeddings,
        "embed_corpus": articles_valid,
    }


def _load_indexes() -> dict:
    """Load all four index files from disk."""
    with open(BM25_FILE,   "rb") as f: bm25           = pickle.load(f)
    with open(BM25_CORPUS, "rb") as f: bm25_corpus    = pickle.load(f)
    embeddings  = np.load(EMBED_FILE)
    with open(EMBED_CORPUS, "rb") as f: embed_corpus  = pickle.load(f)

    print(f"[MiniLM-Indexer] Loaded BM25  ({len(bm25_corpus)} docs) "
          f"+ Embeddings {embeddings.shape}")
    return {
        "bm25":         bm25,
        "bm25_corpus":  bm25_corpus,
        "embeddings":   embeddings,
        "embed_corpus": embed_corpus,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = build_index(force=True)
    print(f"\nBM25 vocab  : {len(result['bm25'].idf)} terms")
    print(f"Corpus size : {len(result['bm25_corpus'])} articles")
    print(f"Embed shape : {result['embeddings'].shape}")
