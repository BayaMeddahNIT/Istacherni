"""
chunk_articles.py
-----------------
Chunking module: converts each raw article dict into a structured chunk
suitable for embedding and storage in ChromaDB.

Each chunk contains:
  - A single text string built from: article text + summary + keywords
  - A metadata dict with all searchable/displayable fields
"""

from typing import List, Dict, Any
import sys
import os

# Allow running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.rag.preprocessing.normalize_arabic import normalize_arabic


def _safe_str(value: Any) -> str:
    """Safely convert a value to string, handling None and lists."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


def article_to_chunk(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single article dict into a chunk dict with:
      - "id"       : unique identifier (from article["id"])
      - "text"     : the text that will be embedded
      - "metadata" : dict of all searchable / displayable fields
    """
    # --- Build the embeddable text ---
    text_original = _safe_str(
        article.get("text", {}).get("original") if isinstance(article.get("text"), dict)
        else article.get("text")
    )
    summary = _safe_str(article.get("summary", ""))
    keywords_list = article.get("keywords", [])
    keywords_str = " ".join(_safe_str(k) for k in keywords_list if k)

    # Combine into one rich text for embedding
    combined_text = f"{text_original} {summary} {keywords_str}".strip()
    normalized_text = normalize_arabic(combined_text)

    # --- Build metadata (all values must be str/int/float/bool for ChromaDB) ---
    # Some articles store classification as a plain string — fall back to empty dict
    classification = article.get("classification", {})
    if not isinstance(classification, dict):
        classification = {}
    source = article.get("source", {}) or {}
    versioning = article.get("versioning", {}) or {}
    relations = article.get("relations", {}) or {}

    metadata = {
        "id": _safe_str(article.get("id", "")),
        "law_domain": _safe_str(article.get("law_domain", "")),
        "law_name": _safe_str(article.get("law_name", "")),
        "book": _safe_str(article.get("book", "")),
        "title": _safe_str(article.get("title", "")),
        "article_number": _safe_str(article.get("article_number", "")),
        "main_category": _safe_str(classification.get("main_category", "")),
        "sub_category": _safe_str(classification.get("sub_category", "")),
        "section": _safe_str(classification.get("section", "")),
        "text_original": text_original,
        "summary": summary,
        "keywords": keywords_str,
        "legal_type": _safe_str(article.get("legal_type", "")),
        "status": _safe_str(versioning.get("status", "")),
        "source_document": _safe_str(source.get("document", "")),
        "source_year": _safe_str(source.get("year", "")),
    }

    # Filter out empty string values to keep metadata clean
    metadata = {k: v for k, v in metadata.items() if v}

    return {
        "id": _safe_str(article.get("id", "")),
        "text": normalized_text,
        "metadata": metadata,
    }


def chunk_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of raw article dicts into a list of chunk dicts.
    Articles with empty IDs or empty text are skipped with a warning.
    """
    chunks = []
    skipped = 0

    for article in articles:
        chunk = article_to_chunk(article)

        if not chunk["id"]:
            print(f"[WARN] Skipping article with no ID: {article}")
            skipped += 1
            continue

        if not chunk["text"]:
            print(f"[WARN] Skipping article {chunk['id']} — no embeddable text found")
            skipped += 1
            continue

        chunks.append(chunk)

    print(f"[INFO] Chunked {len(chunks)} articles ({skipped} skipped)")
    return chunks


if __name__ == "__main__":
    from backend.rag.ingestion.load_articles import load_all_articles

    articles = load_all_articles()
    chunks = chunk_articles(articles)
    print(f"\nSample chunk ID  : {chunks[0]['id']}")
    print(f"Sample chunk text: {chunks[0]['text'][:200]}...")
    print(f"Sample metadata  : {chunks[0]['metadata']}")
