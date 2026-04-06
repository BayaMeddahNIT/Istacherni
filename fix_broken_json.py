"""
fix_broken_json.py
------------------
Fixes all *.json files in dataset/raw that fail json.loads():
  1. Removes trailing commas before } or ]
  2. Flattens nested arrays [[{...}]] → [{...}]
  3. Re-saves as a proper JSON array

Run once: python fix_broken_json.py
"""

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "dataset" / "raw"


def remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } (JavaScript/JSON5 style)."""
    # Handle: ,  \n  } or , \n ]
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text


def flatten_nested_arrays(data):
    """Flatten [[{...}]] → [{...}] recursively."""
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(item)
            elif isinstance(item, dict):
                result.append(item)
        return result
    return data


def try_load(content: str):
    """Try multiple strategies to parse JSON content."""
    # Strategy 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: remove trailing commas
    fixed = remove_trailing_commas(content)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: stream-decode concatenated objects
    decoder = json.JSONDecoder()
    articles = []
    idx = 0
    length = len(fixed)
    while idx < length:
        while idx < length and fixed[idx] in ' \t\r\n[,':
            idx += 1
        if idx >= length or fixed[idx] == ']':
            break
        try:
            obj, end_idx = decoder.raw_decode(fixed, idx)
            if isinstance(obj, dict):
                articles.append(obj)
            elif isinstance(obj, list):
                articles.extend(o for o in obj if isinstance(o, dict))
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1

    return articles if articles else None


def fix_file(path: Path) -> bool:
    """Try to fix a broken JSON file. Returns True if fixed, False if already OK."""
    content = path.read_text(encoding="utf-8")

    # Check if already valid
    try:
        data = json.loads(content)
        data = flatten_nested_arrays(data)
        if isinstance(data, list):
            return False  # Already fine
    except json.JSONDecodeError:
        pass

    # Try to fix it
    data = try_load(content)
    if data is None:
        print(f"  [FAIL] Could not fix: {path.name}")
        return False

    data = flatten_nested_arrays(data)
    if not isinstance(data, list):
        data = [data]

    # Remove non-dict items
    articles = [item for item in data if isinstance(item, dict)]

    if not articles:
        print(f"  [SKIP] No valid articles found in: {path.name}")
        return False

    # Rewrite as a clean JSON array
    fixed_content = json.dumps(articles, ensure_ascii=False, indent=4)
    path.write_text(fixed_content, encoding="utf-8")
    print(f"  [FIXED] {path.name}  ({len(articles)} articles)")
    return True


def main():
    all_json = sorted(RAW_DIR.rglob("*.json"))
    print(f"Scanning {len(all_json)} .json files...\n")

    fixed_count = 0
    ok_count = 0
    fail_count = 0

    for path in all_json:
        # Skip non-article helper files
        if path.name.startswith("add_") or path.name.startswith("test"):
            print(f"  [SKIP] {path.name}  (helper/test file)")
            continue

        try:
            result = fix_file(path)
            if result:
                fixed_count += 1
            else:
                ok_count += 1
                print(f"  [OK]   {path.name}")
        except Exception as e:
            fail_count += 1
            print(f"  [ERR]  {path.name}: {e}")

    print(f"\n{'─'*50}")
    print(f"  Already valid : {ok_count}")
    print(f"  Fixed         : {fixed_count}")
    print(f"  Failed        : {fail_count}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
