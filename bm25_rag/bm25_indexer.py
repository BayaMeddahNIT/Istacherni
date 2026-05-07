"""
bm25_indexer.py
---------------
Builds (and saves) a BM25 index over the entire Algerian law dataset.
Run this ONCE before querying.

Usage:
  python bm25_rag/bm25_indexer.py
  python -m bm25_rag.bm25_indexer

Output:
  bm25_rag/index/bm25_index.pkl   ← serialised BM25Okapi object
  bm25_rag/index/bm25_corpus.pkl  ← list of article dicts (same order as BM25)
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from bm25_rag.bm25_loader import load_all_articles

# ── Index storage ──────────────────────────────────────────────────────────────
INDEX_DIR     = Path(__file__).parent / "index"
BM25_FILE     = INDEX_DIR / "bm25_index.pkl"
CORPUS_FILE   = INDEX_DIR / "bm25_corpus.pkl"


# ── Arabic tokeniser ───────────────────────────────────────────────────────────
def tokenize_arabic(text: str) -> list[str]:
    """
    Lightweight Arabic tokeniser for BM25:
    1. Strip diacritics (tashkeel) and tatweel
    2. Normalise common letter variants  (أ إ آ -> ا, ة -> ه, ى -> ي)
    3. Split on whitespace / punctuation
    4. Discard tokens shorter than 2 characters
    """
    # Remove diacritics (U+064B – U+065F) and tatweel (U+0640)
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)

    # Normalise letter variants
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ة",      "ه", text)
    text = re.sub(r"ى",      "ي", text)

    # Split on anything that is not an Arabic/Latin letter or digit
    tokens = re.split(r"[^\w\u0600-\u06FF]+", text.lower())

    return [t for t in tokens if len(t) >= 2]


def build_document_text(article: dict) -> str:
    """
    Combine several article fields into a rich BM25 document string.
    Keywords are repeated 2x to boost their importance.
    """
    law_name = article.get("law_name", "")
    art_num  = article.get("article_number", "")
    header = f"[{law_name} - المادة {art_num}]" if law_name and art_num else ""
    
    parts = [
        header,
        article.get("title", ""),
        article.get("text_original", ""),
        article.get("summary", ""),
        article.get("legal_conditions_summary", ""),
        article.get("penalties_summary", ""),
        " ".join(article.get("keywords", [])),   # first pass
        " ".join(article.get("keywords", [])),   # second pass (x 2 weight)
    ]
    return " ".join(p for p in parts if p)


def build_index(force: bool = False) -> tuple:
    """
    Build the BM25 index and save it to disk.

    Args:
        force: If True, rebuild even if index already exists.

    Returns:
        (bm25, corpus) tuple.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not force and BM25_FILE.exists() and CORPUS_FILE.exists():
        print("[BM25-Indexer] Index already exists. Loading from disk...")
        return _load_index()

    print("[BM25-Indexer] Loading articles...")
    articles = load_all_articles()

    print("[BM25-Indexer] Tokenising corpus...")
    corpus_texts = [build_document_text(a) for a in articles]
    tokenized = [tokenize_arabic(t) for t in corpus_texts]

    # Filter out empty documents
    valid = [(tok, art) for tok, art in zip(tokenized, articles) if tok]
    tokenized_valid, articles_valid = zip(*valid) if valid else ([], [])

    print(f"[BM25-Indexer] Building BM25Okapi over {len(articles_valid)} docs...")
    bm25 = BM25Okapi(list(tokenized_valid))

    # Persist
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(CORPUS_FILE, "wb") as f:
        pickle.dump(list(articles_valid), f)

    print(f"[BM25-Indexer] [ok] Index saved -> {BM25_FILE}")
    print(f"[BM25-Indexer] [ok] Corpus saved -> {CORPUS_FILE}\n")
    return bm25, list(articles_valid)


def _load_index() -> tuple:
    """Load a previously built index from disk."""
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(CORPUS_FILE, "rb") as f:
        corpus = pickle.load(f)
    print(f"[BM25-Indexer] Loaded {len(corpus)} docs from cached index.")
    return bm25, corpus


if __name__ == "__main__":
    bm25, corpus = build_index(force=True)
    print(f"Corpus size : {len(corpus)} articles")
    print(f"Vocab size  : {len(bm25.idf)} terms")
