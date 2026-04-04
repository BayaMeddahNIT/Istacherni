"""
bm25_loader.py
--------------
Standalone data loader for BM25 RAG.
Reads ALL *.json and *.jsonl files from dataset/raw/** and returns
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
        # Try 'original', then 'content', then 'text', otherwise default to empty string
        return str(text_field.get("original") or text_field.get("content") or text_field.get("text") or "")

    # Structure B: {"text": "..."} (flat string)
    if isinstance(text_field, str):
        return text_field

    # Structure C: {"text_original": "..."} (some files)
    text_original = article.get("text_original")
    if isinstance(text_original, str):
        return text_original

    # Final Safety Net: If it's None or something weird, return empty string
    # This prevents the '.strip()' AttributeError on the calling side.
    return ""


def _normalize_article(raw: dict) -> dict | None:
    """
    Normalise a raw JSON article dict into a canonical BM25 document.
    Returns None if the article has no usable text.
    """
    text = _extract_text(raw).strip()
    if not text:
        return None

    article_num = raw.get("article_number", "")

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


def _load_json_file(path: Path) -> list[dict]:
    """Load a .json or .jsonl file and return a list of raw article dicts.

    Handles three data layouts:
      1. Standard JSON  – a single array ``[{…}, {…}]`` or dict-wrapped list.
      2. True JSONL      – one compact JSON object per line.
      3. Pretty-printed  – multiple JSON objects separated by whitespace
         (the actual format used by this project's .jsonl files).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return []

    # ── 1. Try parsing the whole file as a single JSON value ───────────
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
    except json.JSONDecodeError:
        pass

    # ── 2. Stream-decode concatenated JSON objects ─────────────────────
    #    Works for both true JSONL *and* pretty-printed objects separated
    #    by whitespace / blank lines.
    decoder = json.JSONDecoder()
    articles: list[dict] = []
    idx = 0
    length = len(content)
    while idx < length:
        # Skip whitespace between objects
        while idx < length and content[idx] in ' \t\r\n':
            idx += 1
        if idx >= length:
            break
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            if isinstance(obj, dict):
                articles.append(obj)
            elif isinstance(obj, list):
                articles.extend(o for o in obj if isinstance(o, dict))
            idx = end_idx
        except json.JSONDecodeError:
            # Skip one character and retry (handles stray commas, etc.)
            idx += 1

    if not articles:
        print(f"  [WARN] No JSON objects decoded from {path.name}")
    return articles


def load_all_articles(data_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """
    Walk data_dir recursively and load every *.json / *.jsonl file.
    Returns a deduplicated flat list of normalised article dicts.
    Skip helper scripts and non-article files automatically.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    seen_ids: set[str] = set()
    articles: list[dict] = []

    files = sorted(data_dir.rglob("*.json")) + sorted(data_dir.rglob("*.jsonl"))
    print(f"[BM25-Loader] Found {len(files)} JSON/JSONL files in {data_dir}")

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
            print(f"  ✓ {path.name:45s}  ({file_count} articles)")

    print(f"\n[BM25-Loader] Total unique articles loaded: {len(articles)}\n")
    return articles


if __name__ == "__main__":
    docs = load_all_articles()
    print(f"Sample: {docs[0]['id']} — {docs[0]['title'][:60]}")
