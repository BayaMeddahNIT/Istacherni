"""
qwen_loader.py
--------------
Standalone data loader for Qwen RAG.
Reads ALL *.json files from dataset/raw/** and returns a flat list of
normalised article dicts — NO dependency on bm25_rag or any other package.

Each article dict is guaranteed to have:
  id, law_domain, law_name, article_number, title,
  text_original, summary, keywords, penalties_summary, legal_conditions_summary
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# Walk up from  qwen_rag/  to the project root, then into dataset/raw/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_text(article: dict) -> str:
    """Pull the main Arabic text from various possible JSON structures."""
    text_field = article.get("text")

    # Structure A: {"text": {"original": "..."}} or {"text": {"content": "..."}}
    if isinstance(text_field, dict):
        original = (
            text_field.get("original")
            or text_field.get("content")
            or text_field.get("text")
            or ""
        )
        # Multi-article format: {"original": {"1": "...", "4": "..."}}
        if isinstance(original, dict):
            return " ".join(str(v) for v in original.values() if v)
        return str(original)

    # Structure B: {"text": "..."} — flat string
    if isinstance(text_field, str):
        return text_field

    # Structure C: {"text_original": "..."} — some files use this key
    text_original = article.get("text_original")
    if isinstance(text_original, str):
        return text_original

    return ""


def _normalize_article(raw: dict) -> dict | None:
    """
    Normalise a raw JSON article dict into a canonical document.
    Returns None if the article has no usable text.
    """
    text = _extract_text(raw).strip()
    if not text:
        return None

    art_num_raw = raw.get("article_number", "")
    if isinstance(art_num_raw, list):
        article_num = "-".join(str(x) for x in art_num_raw)
    else:
        article_num = art_num_raw

    return {
        "id":                       raw.get("id") or f"ART_{article_num}",
        "law_domain":               raw.get("law_domain", ""),
        "law_name":                 raw.get("law_name", ""),
        "article_number":           str(article_num),
        "title":                    raw.get("title", ""),
        "text_original":            text,
        "summary":                  raw.get("summary", ""),
        "keywords":                 raw.get("keywords", []),
        "penalties_summary":        raw.get("penalties_summary", ""),
        "legal_conditions_summary": raw.get("legal_conditions_summary", ""),
    }


# ── JSON file loader ───────────────────────────────────────────────────────────

def _remove_trailing_commas(text: str) -> str:
    """Fix trailing commas before ] or } — a common formatting issue."""
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text


def _load_json_file(path: Path) -> list[dict]:
    """
    Load a .json file and return a list of raw article dicts.
    Handles:
      1. Standard JSON array:         [{...}, {...}]
      2. Dict-wrapped list:           {"articles": [{...}]}
      3. Either of the above with trailing commas (auto-fixed)
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return []

    for attempt, text in enumerate([content, _remove_trailing_commas(content)]):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                # Flatten [[{...}]] to [{...}]
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
                continue  # try again with trailing commas removed
            print(f"  [WARN] Could not parse {path.name} as JSON.")
    return []


# ── Public API ─────────────────────────────────────────────────────────────────

def load_all_articles(data_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """
    Walk data_dir recursively, load every *.json file, and return a
    deduplicated flat list of normalised article dicts.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Make sure the project root contains  dataset/raw/"
        )

    seen_ids: set[str] = set()
    articles: list[dict] = []

    files = sorted(data_dir.rglob("*.json"))
    print(f"[Qwen-Loader] Found {len(files)} JSON files in {data_dir}")

    for path in files:
        # Skip known non-article helper scripts
        if path.name.startswith("add_") or path.name.startswith("test"):
            continue

        raw_list   = _load_json_file(path)
        file_count = 0

        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            doc = _normalize_article(raw)
            if doc is None:
                continue
            if doc["id"] in seen_ids:
                continue
            seen_ids.add(doc["id"])
            articles.append(doc)
            file_count += 1

        if file_count:
            print(f"  [ok] {path.name:50s}  ({file_count} articles)")

    print(f"\n[Qwen-Loader] Total unique articles loaded: {len(articles)}\n")
    return articles


if __name__ == "__main__":
    docs = load_all_articles()
    if docs:
        print(f"Sample: {docs[0]['id']} -- {docs[0]['title'][:60]}")