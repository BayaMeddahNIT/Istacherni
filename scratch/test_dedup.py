
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from bm25_rag.bm25_loader import _normalize_article, load_all_articles

def test_deduplication_logic():
    # Mock articles
    raw_articles = [
        {
            "id": "ART_1",
            "article_number": "1",
            "text": "Original text A",
            "text_explanation": "Explanation A"
        },
        {
            "id": "ART_1",
            "article_number": "1",
            "text": "Original text A",
            "text_explanation": "Explanation A"
        }, # Exact duplicate
        {
            "id": "ART_1",
            "article_number": "1",
            "text": "Original text B",
            "text_explanation": "Explanation A"
        }, # Same ID, different text
        {
            "id": "ART_1",
            "article_number": "1",
            "text": "Original text A",
            "text_explanation": "Explanation B"
        }, # Same ID, different explanation
    ]

    # Current logic (as of now)
    seen_ids = set()
    processed = []
    for raw in raw_articles:
        doc = _normalize_article(raw)
        if doc and doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            processed.append(doc)
    
    print(f"Current logic results: {[p['id'] for p in processed]}")
    # Expected: ['ART_1'] - It skips all others because of ID match.

    # Proposed logic
    # (We will implement this in bm25_loader.py, but let's test it here)
    id_versions = {} # original_id -> list of (text_original, text_explanation)
    processed_new = []
    
    for raw in raw_articles:
        doc = _normalize_article(raw)
        if not doc: continue
        
        orig_id = doc["id"]
        text = doc["text_original"]
        expl = doc["text_explanation"]
        
        if orig_id not in id_versions:
            id_versions[orig_id] = [(text, expl)]
            processed_new.append(doc)
        else:
            # Check if this exact content was already seen for this ID
            is_duplicate = False
            for seen_text, seen_expl in id_versions[orig_id]:
                if text == seen_text and expl == seen_expl:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Same ID, different content -> Suffix it
                version_count = len(id_versions[orig_id])
                doc["id"] = f"{orig_id}_{version_count}"
                id_versions[orig_id].append((text, expl))
                processed_new.append(doc)
    
    print(f"Proposed logic results: {[p['id'] for p in processed_new]}")
    # Expected: ['ART_1', 'ART_1_1', 'ART_1_2']

if __name__ == "__main__":
    test_deduplication_logic()
