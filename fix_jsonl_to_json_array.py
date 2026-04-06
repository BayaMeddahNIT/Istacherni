"""
Converts all .jsonl files in dataset/raw (recursively) from the current format
(JSON objects separated by blank lines) into proper JSON arrays:
  [ { ... }, { ... }, ... ]
The files are modified in-place.
"""

import os
import re

RAW_DIR = os.path.join(os.path.dirname(__file__), "dataset", "raw")

def convert_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove leading/trailing whitespace from the whole file
    content = content.strip()

    # The objects are separated by one or more blank lines.
    # We need to insert a comma between consecutive objects: between "}" and "{"
    # Pattern: closing brace (possibly with trailing whitespace/newlines)
    # followed by one or more blank lines, then opening brace
    # Replace: }  \n\n  { -> },\n    {
    converted = re.sub(r'\}\s*\n\s*\n\s*\{', '},\n    {', content)

    # Wrap in a JSON array
    result = "[\n    " + converted + "\n]\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result)

    return True

def main():
    files_processed = []
    for root, dirs, files in os.walk(RAW_DIR):
        for fname in files:
            if fname.endswith(".jsonl"):
                fpath = os.path.join(root, fname)
                convert_file(fpath)
                files_processed.append(fpath)
                print(f"  ✓ Converted: {os.path.relpath(fpath, RAW_DIR)}")

    print(f"\nDone! Processed {len(files_processed)} file(s).")

if __name__ == "__main__":
    main()
