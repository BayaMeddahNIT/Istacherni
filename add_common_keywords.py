"""
add_common_keywords.py
──────────────────────
Enriches ALL articles in dataset/raw/** with "common-language" Arabic keywords
so that BGE-M3 retrieval works better for everyday user queries.

For each article:
  - Reads existing keywords (technical terms kept as-is)
  - Calls Gemini to suggest 3-6 simple, everyday Arabic keywords
  - Appends only truly new ones (no duplicates)
  - Saves the file back in-place (JSON pretty-printed, UTF-8)

Usage:
    python add_common_keywords.py
    python add_common_keywords.py --dry-run      # preview without saving
    python add_common_keywords.py --file Okod    # process only files matching name
"""

import sys
import io
import json
import time
import re
import argparse
from pathlib import Path

# Force UTF-8 output (avoids cp1252 emoji crash on Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
import os

try:
    from google import genai
    from google.genai import types as genai_types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai_old
    USE_NEW_SDK = False

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"

load_dotenv(PROJECT_ROOT / ".env")

# All available keys — rotated on 429 quota errors
ALL_KEYS = [k for k in [
    os.getenv("EVAL_GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY"),
    os.getenv("BM25_GEMINI_API_KEY"),
    os.getenv("AGENTIC_GEMINI_API_KEY"),
    os.getenv("GRAPH_GEMINI_API_KEY"),
] if k]
if not ALL_KEYS:
    sys.exit("No Gemini API key found in .env")

MODEL_NAME  = "gemini-2.0-flash-lite"  # higher free-tier RPM than 2.0-flash
BATCH_SIZE  = 5       # articles per Gemini call
SLEEP_SEC   = 2.0    # polite pause between calls
MAX_RETRIES = 3       # retries per batch before giving up

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "anta musa'id qanuni mutakhassiss fi al-qanun al-jaza'iri.\n"
    "mahammatuka: li kull madda qanuniyya, iqtarih kalimat miftahiyya basita "
    "yastakhdimaha al-'amma fi yawmiyyatihim.\n"
    "al-kalimat yajib an:\n"
    "- takun bi-l-'arabiyya al-fus-ha al-basita aw al-darija\n"
    "- tu'abbir 'amma yabhathu 'anhu al-muwatin al-'adi\n"
    "- la tukarrir al-kalimat al-mawjuda misbaqan\n"
    "- takun bayna 3 wa 6 kalimat/'ibarat muqtasara\n"
    "ajib faqat bi-qa'imat JSON salihah bidun ayy nass idafi.\n"
)


def _make_prompt(articles: list) -> str:
    lines = []
    for i, a in enumerate(articles):
        existing = a.get("keywords") or a.get("tags") or []
        if isinstance(existing, list):
            existing_str = " | ".join(existing)
        else:
            existing_str = str(existing)
        explanation = (
            a.get("text_explanation")
            or a.get("definition")
            or a.get("summary")
            or (a.get("text") or {}).get("original", "")
            or ""
        )
        lines.append(
            f"Article {i+1}:\n"
            f"  Title: {a.get('title', '')}\n"
            f"  Text: {str(explanation)[:300]}\n"
            f"  Existing keywords: {existing_str}\n"
        )
    return (
        "Suggest common-language Arabic keywords for each article.\n"
        "Return a JSON list of lists, one sub-list per article.\n"
        "Example: [[\"keyword1\", \"keyword2\"], [\"keyword3\", \"keyword4\"]]\n\n"
        + "\n".join(lines)
    )


# Mutable key-rotation state
_key_index = [0]


def _get_client():
    """Return a fresh client using the current key."""
    key = ALL_KEYS[_key_index[0] % len(ALL_KEYS)]
    if USE_NEW_SDK:
        return genai.Client(api_key=key)
    else:
        genai_old.configure(api_key=key)
        return genai_old.GenerativeModel(MODEL_NAME)


def _rotate_key():
    _key_index[0] += 1
    new_key_index = _key_index[0] % len(ALL_KEYS)
    print(f"  [KEY] Rotating to key #{new_key_index + 1}/{len(ALL_KEYS)}")


def _call_gemini(articles: list) -> list:
    """Call Gemini with automatic key rotation on 429 errors."""
    prompt = _make_prompt(articles)
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    for attempt in range(MAX_RETRIES * len(ALL_KEYS)):
        client = _get_client()
        try:
            if USE_NEW_SDK:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    ),
                )
                raw = response.text.strip()
            else:
                response = client.generate_content(
                    [SYSTEM_PROMPT, prompt],
                    generation_config={"temperature": 0.3, "max_output_tokens": 1024},
                )
                raw = response.text.strip()

            # Strip markdown code fences
            m = re.search(r"```(?:json)?\s*(\[[\s\S]*\])\s*```", raw)
            if m:
                raw = m.group(1)
            else:
                start = raw.find("[")
                end   = raw.rfind("]")
                if start != -1 and end != -1:
                    raw = raw[start : end + 1]

            parsed = json.loads(raw)
            if parsed and not isinstance(parsed[0], list):
                return [parsed]  # single-article flat list
            return parsed

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 5 + attempt * 3
                print(f"  [QUOTA] Key exhausted (attempt {attempt+1}), rotating key + waiting {wait}s...")
                _rotate_key()
                time.sleep(wait)
            else:
                print(f"  [WARN] Gemini error: {e} — skipping batch")
                return [[] for _ in articles]

    print(f"  [SKIP] All keys exhausted for this batch — skipping")
    return [[] for _ in articles]


def _get_keyword_field(article: dict) -> str:
    if "keywords" in article:
        return "keywords"
    if "tags" in article:
        return "tags"
    return "keywords"


def _merge_keywords(existing, new_ones: list) -> list:
    if isinstance(existing, str):
        existing = [existing]
    result = list(existing or [])
    existing_lower = {k.strip().lower() for k in result}
    for kw in new_ones:
        kw = kw.strip()
        if kw and kw.lower() not in existing_lower:
            result.append(kw)
            existing_lower.add(kw.lower())
    return result


def _load_json_file(path: Path) -> list:
    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        content = re.sub(r",\s*(\})", r"\1", content)
        content = re.sub(r",\s*(\])", r"\1", content)
        data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return []


def process_all(dry_run: bool = False, target_file: str = None, resume: bool = False):
    files = sorted(RAW_DATA_DIR.rglob("*.json"))
    files = [f for f in files if not f.name.startswith(("add_", "test"))]
    if target_file:
        files = [f for f in files if target_file.lower() in f.stem.lower()]

    total_updated = 0
    total_articles = 0

    for path in files:
        print(f"\n{'-'*60}")
        print(f"[FILE] {path.relative_to(PROJECT_ROOT)}")

        try:
            articles = _load_json_file(path)
        except Exception as e:
            print(f"  [ERROR] Cannot read: {e}")
            continue

        if not articles:
            print("  (empty)")
            continue

        # --resume: skip files that already have common-language kws appended
        if resume:
            first_kws = (articles[0].get("keywords") or articles[0].get("tags") or [])
            if len(first_kws) > 6:  # already enriched (original max ~6 technical kws)
                print("  [SKIP] Already enriched")
                continue

        changed = False

        for batch_start in range(0, len(articles), BATCH_SIZE):
            batch = articles[batch_start : batch_start + BATCH_SIZE]
            suggestions = _call_gemini(batch)

            for i, article in enumerate(batch):
                if not isinstance(article, dict):
                    continue
                new_kws = suggestions[i] if i < len(suggestions) else []
                if not new_kws:
                    continue

                field  = _get_keyword_field(article)
                old    = article.get(field) or []
                merged = _merge_keywords(old, new_kws)
                added  = [k for k in merged if k not in (old or [])]

                if added:
                    if not dry_run:
                        article[field] = merged
                    total_updated += 1
                    changed = True
                    art_id = article.get("id") or article.get("article_number", "?")
                    print(f"  [OK] {art_id} +{len(added)} kws: {added}")

            total_articles += len(batch)
            time.sleep(SLEEP_SEC)

        if changed and not dry_run:
            path.write_text(
                json.dumps(articles, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            print(f"  [SAVED] {path.name}")

    print(f"\n{'='*60}")
    print(f"Done -- {total_updated}/{total_articles} articles enriched")
    if dry_run:
        print("   (DRY RUN -- no files were modified)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add common-language keywords to all articles")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--file",    type=str, default=None, help="Only process files matching this name")
    parser.add_argument("--resume",  action="store_true", help="Skip files already enriched")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN mode -- files will NOT be modified\n")
    print(f"Using {len(ALL_KEYS)} API keys, model: {MODEL_NAME}\n")

    process_all(dry_run=args.dry_run, target_file=args.file, resume=args.resume)
