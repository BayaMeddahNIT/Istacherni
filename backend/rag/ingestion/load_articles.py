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

    # Deduplicate by hashing the full article content (excluding _source_file)
    # This catches articles from different books that share the same ID
    seen_hashes: set = set()
    unique_articles = []

    for art in articles:
        art_hash = hashlib.md5(
            json.dumps(
                {k: v for k, v in art.items() if k != "_source_file"},
                sort_keys=True,
                ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

        if art_hash not in seen_hashes:
            seen_hashes.add(art_hash)
            unique_articles.append(art)

    duplicates = len(articles) - len(unique_articles)
    if duplicates:
        print(f"[INFO] Removed {duplicates} duplicate articles (identical content across files)")
    print(f"[INFO] Unique articles: {len(unique_articles)}")
    return unique_articles


if __name__ == "__main__":
    arts = load_all_articles()
    print(f"\nSample article ID: {arts[0].get('id', 'N/A')}")
    print(f"Sample law domain: {arts[0].get('law_domain', 'N/A')}")