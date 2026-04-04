"""
load_articles.py
----------------
Ingestion module: walks all subdirectories of dataset/raw/ and loads
every .json and .jsonl file.
  - .json  files are expected to be a JSON array of article dicts.
  - .jsonl files are expected to have one JSON object per line (JSON Lines).

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


def _load_file(path: Path) -> List[Dict[str, Any]]:
    """Load a .json or .jsonl file and return a list of article dicts."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Could not read {path.name}: {e}")
        return []

    # JSONL format: one JSON object per line
    if path.suffix == ".jsonl":
        articles = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    articles.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name} line {line_num}: JSON parse error: {e}")
        return articles

    # Regular JSON: expect a list of dicts
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some files wrap a list under a key
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse {path.name}: {e}")
    return []


def load_all_articles(dataset_dir: Path = DATASET_DIR) -> List[Dict[str, Any]]:
    """
    Walk all subdirectories inside `dataset_dir` and load every .json / .jsonl file.

    Returns a flat list of all article dicts, deduplicated by full content.
    """
    articles: List[Dict[str, Any]] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_files = sorted(dataset_dir.rglob("*.json")) + sorted(dataset_dir.rglob("*.jsonl"))

    if not data_files:
        raise FileNotFoundError(f"No .json or .jsonl files found under: {dataset_dir}")

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