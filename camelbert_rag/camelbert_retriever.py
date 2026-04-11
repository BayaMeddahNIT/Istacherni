from __future__ import annotations

from typing import Any
import numpy as np

# This now returns a single Chroma Collection object
from camelbert_rag.camelbert_indexer import build_index     
from camelbert_rag.camelbert_embedder import embed_texts

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
        # In Chroma, 'distances' for cosine space are usually 1 - similarity 
        # or just similarity depending on version. We'll treat it as score.
        score = 1 - results['distances'][0][i] 
        
        if score < score_threshold:
            continue

        metadata = results['metadatas'][0][i]
        text = results['documents'][0][i]

        hits.append(
            {
                "id":               results['ids'][0][i],
                "law_name":         metadata.get("law_name", "قانون جزائري"),
                "law_domain":       metadata.get("law_domain", ""),
                "article_number":   metadata.get("article_number", ""),
                "title":            metadata.get("title", ""),
                "text_original":    text,
                "penalties_summary": metadata.get("penalties_summary", ""),
                "legal_conditions_summary": metadata.get("legal_conditions_summary", ""),
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