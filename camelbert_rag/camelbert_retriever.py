from __future__ import annotations

from typing import Any
import numpy as np

# This now returns a single Chroma Collection object
from camelbert_rag.camelbert_indexer import build_index     
from camelbert_rag.camelbert_embedder import embed_texts
import re

def normalize_for_camelbert(text: str) -> str:
    if not text: return ""
    # Eastern Arabic → Western Arabic numerals
    eastern = '٠١٢٣٤٥٦٧٨٩'
    western = '0123456789'
    trans = str.maketrans(eastern, western)
    text = text.translate(trans)
    
    # Normalize article reference format
    text = re.sub(r'المادة\s+(\d+)', r'المادة \1', text)
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    
    # Normalize Alef, Ya, Ta Marbuta
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ى', 'ي')
    text = text.replace('ة', 'ه')
    
    return text

# ── Lazy-loaded singleton ──────────────────────────────────────────────────────
_collection = None

def _get_index():
    global _collection
    if _collection is None:
        # Fixed: We only expect one return value now (the Chroma Collection)
        _collection = build_index()   
    return _collection


# ── Public API ─────────────────────────────────────────────────────────────────

def camelbert_retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-K most semantically relevant law articles for *query*
    using CAMeLBERT dense embeddings + ChromaDB.
    """
    collection = _get_index()

    # Normalize query symmetrically with the indexer
    query = normalize_for_camelbert(query)

    # Embed query (using your existing embedder)
    # Chroma prefers a list of embeddings
    q_vec = embed_texts([query], normalize=True).tolist()

    # Chroma search
    results = collection.query(
        query_embeddings=q_vec,
        n_results=top_k
    )

    hits: list[dict[str, Any]] = []

    # Chroma returns lists of lists (because you can batch queries)
    # We take the 0th index because we only have one query
    for i in range(len(results['ids'][0])):
        # In Chroma, distances for cosine space = 1 - similarity
        score = 1 - results['distances'][0][i]

        if score < score_threshold:
            continue

        doc_id   = results['ids'][0][i]
        metadata = results['metadatas'][0][i]
        text     = results['documents'][0][i]

        # ── Robust article_number extraction ──────────────────────────────
        # ChromaDB may return an empty string if the source article had no
        # article_number set.  Fall back to parsing the doc id (art_N) so
        # that source strings are never written as "law_name - المادة  ".
        article_number = metadata.get("article_number", "") or ""
        if not article_number.strip():
            # Try to find a number in the document text (e.g. "المادة 219")
            import re as _re
            m = _re.search(r'المادة\s+(\S+)', text[:200])
            if m:
                article_number = m.group(1)
            else:
                article_number = doc_id  # fallback to the Chroma doc id

        law_name = metadata.get("law_name", "") or "قانون جزائري"

        hits.append(
            {
                "id":               doc_id,
                "law_name":         law_name,
                "law_domain":       metadata.get("law_domain", ""),
                "article_number":   article_number,
                "title":            metadata.get("title", ""),
                "text_original":    text,
                "penalties_summary":           metadata.get("penalties_summary", ""),
                "legal_conditions_summary":    metadata.get("legal_conditions_summary", ""),
                "keywords":         metadata.get("keywords", []),
                "score":            round(score, 4),
            }
        )

    # Sort by score descending
    hits.sort(key=lambda r: r["score"], reverse=True)
    return hits


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
        "عقوبة تزوير الوثائق الرسمية",
    ]

    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {q}")
        print(f"{'='*65}")
        hits = camelbert_retrieve(q, top_k=3, score_threshold=0.0)
        if not hits:
            print("   No results.")
        for i, r in enumerate(hits, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Title : {r['title']}")
            print(f"       Score : {r['score']}")
            print(f"       Text  : {r['text_original'][:120]}…")