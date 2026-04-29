from typing import List, Dict, Any
import sys
import os
import hashlib  # Added for content verification

# Allow running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.rag.preprocessing.normalize_arabic import normalize_arabic

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)

def article_to_chunk(article: Dict[str, Any]) -> Dict[str, Any]:
    # --- Build the embeddable text ---
    text_original = _safe_str(
        article.get("text", {}).get("original") if isinstance(article.get("text"), dict)
        else article.get("text")
    )
    summary = _safe_str(article.get("summary", ""))
    keywords_list = article.get("keywords", [])
    keywords_str = " ".join(_safe_str(k) for k in keywords_list if k)

    combined_text = f"{text_original} {summary} {keywords_str}".strip()
    normalized_text = normalize_arabic(combined_text)

    # --- Build metadata ---
    classification = article.get("classification", {})
    if not isinstance(classification, dict):
        classification = {}
    source = article.get("source", {}) or {}
    versioning = article.get("versioning", {}) or {}

    metadata = {
        "id": _safe_str(article.get("id", "")),
        "law_domain": _safe_str(article.get("law_domain", "")),
        "law_name": _safe_str(article.get("law_name", "")),
        "article_number": _safe_str(article.get("article_number", "")),
        "main_category": _safe_str(classification.get("main_category", "")),
        "sub_category": _safe_str(classification.get("sub_category", "")),
        "text_original": text_original,
        "summary": summary,
        "status": _safe_str(versioning.get("status", "")),
    }

    metadata = {k: v for k, v in metadata.items() if v}

    return {
        "id": _safe_str(article.get("id", "")),
        "text": normalized_text,
        "metadata": metadata,
    }

def chunk_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    skipped = 0
    unique_chunks = []
    for article in articles:
        chunk = article_to_chunk(article)
        if not chunk["id"] or not chunk["text"]:
            skipped += 1
            continue
        unique_chunks.append(chunk)

    print(f"[INFO] Processed {len(unique_chunks)} chunks.")
    if skipped:
        print(f"[INFO] Skipped {skipped} invalid articles.")
    
    return unique_chunks