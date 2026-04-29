import json
import os
from pathlib import Path
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"

def _extract_content(art):
    """Extracts text_original and text_explanation for comparison."""
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
    
    return text_original, text_explanation

def find_duplicates():
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return

    # Map: article_id -> list of (file_path, text_original, text_explanation)
    id_map = defaultdict(list)
    
    files = sorted(DATASET_DIR.rglob("*.json"))
    print(f"Scanning {len(files)} JSON files...\n")

    for path in files:
        if path.name.startswith("add_") or path.name.startswith("test"):
            continue
            
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            
            # Normalize list format
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
                else:
                    data = [data]
            
            if not isinstance(data, list):
                continue
                
            for art in data:
                if not isinstance(art, dict): continue
                
                art_id = str(art.get("id") or "").strip()
                if not art_id: continue
                
                text_orig, text_expl = _extract_content(art)
                relative_path = path.relative_to(PROJECT_ROOT)
                
                id_map[art_id].append({
                    "path": relative_path,
                    "text": text_orig,
                    "expl": text_expl
                })
                
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Analysis
    true_duplicates = [] # Same ID, Same Content
    different_versions = [] # Same ID, Different Content

    for art_id, occurrences in id_map.items():
        if len(occurrences) > 1:
            # Check unique contents
            seen_contents = []
            for occ in occurrences:
                content_key = (occ["text"], occ["expl"])
                
                found_match = False
                for seen in seen_contents:
                    if seen["key"] == content_key:
                        seen["paths"].append(occ["path"])
                        found_match = True
                        break
                
                if not found_match:
                    seen_contents.append({
                        "key": content_key,
                        "paths": [occ["path"]]
                    })
            
            # If we have multiple paths for the SAME content -> True Duplicates
            for content in seen_contents:
                if len(content["paths"]) > 1:
                    true_duplicates.append({
                        "id": art_id,
                        "paths": content["paths"]
                    })
            
            # If we have multiple UNIQUE contents for the same ID -> Different Versions
            if len(seen_contents) > 1:
                different_versions.append({
                    "id": art_id,
                    "versions": len(seen_contents),
                    "details": seen_contents
                })

    # Report
    print("=" * 60)
    print("DUPLICATE REPORT")
    print("=" * 60)
    
    if not true_duplicates:
        print("No exact duplicates (same ID + same content) found.")
    else:
        print(f"Found {len(true_duplicates)} cases of exact duplicates:\n")
        for dup in true_duplicates:
            print(f"ID: {dup['id']}")
            for p in dup["paths"]:
                print(f"  - {p}")
            print()

    print("-" * 60)
    if not different_versions:
        print("No cases of same ID with different content found.")
    else:
        print(f"Found {len(different_versions)} articles with same ID but DIFFERENT content (these will be suffixed _1, _2, etc.):\n")
        for ver in different_versions:
            print(f"ID: {ver['id']} ({ver['versions']} versions)")
            for idx, content in enumerate(ver['details']):
                print(f"  Version {idx + 1} found in:")
                for p in content["paths"]:
                    print(f"    - {p}")
            print()

    print("=" * 60)

if __name__ == "__main__":
    find_duplicates()
