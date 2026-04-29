"""
load_articles.py
----------------
Ingestion module: walks all subdirectories of dataset/raw/ and loads
every .json file.
  - All .json files must be a JSON array of article dicts: [{...}, {...}, ...]

Returns a flat list of article dicts with all fields preserved.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any


# Resolve dataset root relative to this file's location
# backend/rag/ingestion/load_articles.py → project root → dataset/raw
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = _PROJECT_ROOT / "dataset" / "raw"


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } to recover from minor JSON formatting issues."""
    import re
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text


def _load_file(path: Path) -> List[Dict[str, Any]]:
    """Load a .json file (JSON array format) and return a list of article dicts."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Could not read {path.name}: {e}")
        return []

    # Primary: parse as JSON array
    for attempt, text in enumerate([content, _remove_trailing_commas(content)]):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                # Flatten nested arrays [[{...}]] → [{...}]
                while len(data) == 1 and isinstance(data[0], list):
                    data = data[0]
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [item for item in v if isinstance(item, dict)]
                return [data]
        except json.JSONDecodeError:
            if attempt == 0:
                continue
            print(f"[ERROR] Failed to parse {path.name}")
    return []


def load_all_articles(dataset_dir: Path = DATASET_DIR) -> List[Dict[str, Any]]:
    """
    Walk all subdirectories inside `dataset_dir` and load every .json / .jsonl file.

    Returns a flat list of all article dicts, deduplicated by full content.
    """
    articles: List[Dict[str, Any]] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_files = sorted(dataset_dir.rglob("*.json"))

    if not data_files:
        raise FileNotFoundError(f"No .json files found under: {dataset_dir}")

    for file_path in data_files:
        # Skip helper scripts stored alongside the data
        if file_path.name.startswith("add_") or file_path.name.startswith("test"):
            continue

        data = _load_file(file_path)

        if not data:
            print(f"[WARN] No articles loaded from {file_path.name}")
            continue

        # Tag each article with its source file path for debugging
        for article in data:
            if isinstance(article, dict):
                article.setdefault("_source_file", str(file_path.relative_to(_PROJECT_ROOT)))

        print(f"[INFO] Loaded {len(data):>4} articles from {file_path.relative_to(dataset_dir)}")
        articles.extend(data)

    print(f"\n[INFO] Total articles loaded: {len(articles)}")

    # tracks { "original_id": [(text_original, text_explanation), ...] }
    id_versions: Dict[str, List[tuple]] = {}
    unique_articles = []

    for art in articles:
        # Extract fields for comparison (similar to bm25_loader logic)
        # text_original
        text_field = art.get("text")
        if isinstance(text_field, dict):
            text_original = text_field.get("original") or text_field.get("content") or text_field.get("text") or ""
            if isinstance(text_original, dict):
                text_original = " ".join(str(v) for v in text_original.values() if v)
        elif isinstance(text_field, str):
            text_original = text_field
        else:
            text_original = art.get("text_original") or ""
        
        text_original = str(text_original).strip()

        # text_explanation
        text_explanation = (
            art.get("text_explanation")
            or art.get("definition")
            or art.get("summary")
            or ""
        )
        if isinstance(text_explanation, dict):
            text_explanation = " ".join(str(v) for v in text_explanation.values() if v)
        
        text_explanation = str(text_explanation).strip()

        orig_id = str(art.get("id") or "").strip()
        if not orig_id:
            # If no ID, use a hash or skip? Let's skip for consistency with BM25
            continue

        if orig_id not in id_versions:
            id_versions[orig_id] = [(text_original, text_explanation)]
            unique_articles.append(art)
        else:
            # Check if this exact content was already seen for this ID
            is_duplicate = False
            for seen_text, seen_expl in id_versions[orig_id]:
                if text_original == seen_text and text_explanation == seen_expl:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Same ID, DIFFERENT content -> Create new suffix
                version_count = len(id_versions[orig_id])
                art["id"] = f"{orig_id}_{version_count}"
                id_versions[orig_id].append((text_original, text_explanation))
                unique_articles.append(art)

    print(f"[INFO] Unique articles (after ID-based deduplication): {len(unique_articles)}")
    return unique_articles


if __name__ == "__main__":
    arts = load_all_articles()
    print(f"\nSample article ID: {arts[0].get('id', 'N/A')}")
    print(f"Sample law domain: {arts[0].get('law_domain', 'N/A')}")