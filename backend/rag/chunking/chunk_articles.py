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
    raw_chunks = []
    skipped = 0

    for article in articles:
        chunk = article_to_chunk(article)
        if not chunk["id"] or not chunk["text"]:
            skipped += 1
            continue
        raw_chunks.append(chunk)

    # --- Content-Aware Deduplication Logic ---
    unique_chunks = []
    # tracks { "ID": [list of hashes for this ID] }
    id_content_map = {} 
    # tracks { "ID": counter_integer }
    id_counters = {} 

    for chunk in raw_chunks:
        base_id = chunk["id"]
        # Create a hash of the content to see if it's truly a duplicate
        content_hash = hashlib.md5(chunk["text"].encode('utf-8')).hexdigest()

        if base_id not in id_content_map:
            # First time seeing this ID
            id_content_map[base_id] = [content_hash]
            id_counters[base_id] = 1
            chunk["id"] = f"{base_id}_01"
            unique_chunks.append(chunk)
        else:
            # ID exists, check if content is new
            if content_hash in id_content_map[base_id]:
                # True duplicate (same ID, same content) -> Skip
                continue
            else:
                # Same ID, DIFFERENT content -> Create new suffix
                id_counters[base_id] += 1
                new_count = id_counters[base_id]
                id_content_map[base_id].append(content_hash)
                
                # Update the chunk ID to be unique (e.g., DZ_ART_08_02)
                chunk["id"] = f"{base_id}_{new_count:02d}"
                unique_chunks.append(chunk)

    print(f"[INFO] Processed {len(unique_chunks)} unique content-chunks.")
    print(f"[INFO] Skipped {skipped} invalid and {len(raw_chunks) - len(unique_chunks)} true duplicates.")
    
    return unique_chunks