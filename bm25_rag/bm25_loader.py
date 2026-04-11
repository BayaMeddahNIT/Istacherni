"""
bm25_loader.py
--------------
Standalone data loader for BM25 RAG.
Reads ALL *.json files from dataset/raw/** (JSON arrays) and returns
a flat list of article dicts — completely independent of the rest of the project.

Each article dict is guaranteed to have:
  id, law_domain, law_name, article_number, title,
  text_original, summary, keywords, penalties_summary, legal_conditions_summary
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"




def _extract_text(article: dict) -> str:
    """Pull the main Arabic text from various possible JSON structures."""
    text_field = article.get("text")

    # Structure A: {"text": {"original": "..."}} or {"text": {"content": "..."}}
    if isinstance(text_field, dict):
        original = text_field.get("original") or text_field.get("content") or text_field.get("text") or ""
        # Multi-article format: {"original": {"1": "...", "4": "..."}}
        if isinstance(original, dict):
            return " ".join(str(v) for v in original.values() if v)
        return str(original)

    # Structure B: {"text": "..."} (flat string)
    if isinstance(text_field, str):
        return text_field

    # Structure C: {"text_original": "..."} (some files)
    text_original = article.get("text_original")
    if isinstance(text_original, str):
        return text_original

    return ""


def _normalize_article(raw: dict) -> dict | None:
    """
    Normalise a raw JSON article dict into a canonical BM25 document.
    Returns None if the article has no usable text.
    """
    text = _extract_text(raw).strip()
    if not text:
        return None

    # article_number may be an int, str, or list
    art_num_raw = raw.get("article_number", "")
    if isinstance(art_num_raw, list):
        article_num = "-".join(str(x) for x in art_num_raw)
    else:
        article_num = art_num_raw

    # text_explanation: prefer dedicated field, fall back to definition/summary
    text_explanation = (
        raw.get("text_explanation")
        or raw.get("definition")
        or raw.get("summary")
        or ""
    )
    if isinstance(text_explanation, dict):
        text_explanation = " ".join(str(v) for v in text_explanation.values() if v)

    # keywords: some files use 'tags' instead of 'keywords'
    keywords = raw.get("keywords") or raw.get("tags") or []

    return {
        "id":                       raw.get("id") or f"ART_{article_num}",
        "law_domain":               raw.get("law_domain") or raw.get("law_type", ""),
        "law_name":                 raw.get("law_name") or raw.get("law_type", ""),
        "article_number":           str(article_num),
        "title":                    raw.get("title", ""),
        "text_original":            text,
        "text_explanation":         str(text_explanation).strip(),
        "summary":                  raw.get("summary", ""),
        "keywords":                 keywords,
        "penalties_summary":        raw.get("penalties_summary", ""),
        "legal_conditions_summary": raw.get("legal_conditions_summary", ""),
    }


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } (common JSON formatting issue)."""
    import re
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text


def _load_json_file(path: Path) -> list[dict]:
    """Load a .json file (JSON array format) and return a list of raw article dicts.

    Handles these data layouts:
      1. Standard JSON array: ``[{…}, {…}]``
      2. Dict-wrapped list: ``{"articles": [{…}]}``
      3. Same as above but with trailing commas (auto-fixed)
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
                # Flatten [[{...}]] → [{...}]
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
            print(f"  [WARN] No JSON objects decoded from {path.name}")
    return []


def load_all_articles(data_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """
    Walk data_dir recursively and load every *.json file.
    Returns a deduplicated flat list of normalised article dicts.
    Skip helper scripts and non-article files automatically.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    seen_ids: set[str] = set()
    articles: list[dict] = []

    files = sorted(data_dir.rglob("*.json"))
    print(f"[BM25-Loader] Found {len(files)} JSON files in {data_dir}")

    for path in files:
        # Skip known non-article helper scripts stored in the dataset folder
        if path.name.startswith("add_") or path.name.startswith("test"):
            continue

        raw_list = _load_json_file(path)
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
            print(f"  [ok] {path.name:45s}  ({file_count} articles)")

    print(f"\n[BM25-Loader] Total unique articles loaded: {len(articles)}\n")
    return articles


if __name__ == "__main__":
    docs = load_all_articles()
    print(f"Sample: {docs[0]['id']} — {docs[0]['title'][:60]}")
