"""
qwen_retriever.py
-----------------
Dense retrieval using Qwen3-Embedding-8B + ChromaDB.
Drop-in replacement for bm25_retrieve() — same return schema.

Usage (standalone):
    python qwen_rag/qwen_retriever.py

Programmatic:
    from qwen_rag.qwen_retriever import qwen_retrieve
    results = qwen_retrieve("ما هي عقوبة السرقة؟", top_k=5)

Each result dict contains:
    id, law_name, law_domain, article_number, title,
    text_original, penalties_summary, legal_conditions_summary,
    keywords (list), score (cosine similarity, 0–1)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# ── Lazy singletons ────────────────────────────────────────────────────────────
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        from qwen_rag.qwen_indexer import load_collection
        _collection = load_collection()
    return _collection


# ── Legal Vocabulary Normalizer ────────────────────────────────────────────────
_LEGAL_SYNONYMS: dict[str, str] = {
    "فصل من العمل":    "التسريح التعسفي",
    "الفصل من العمل":  "التسريح التعسفي",
    "طردت من العمل":   "التسريح التعسفي",
    "طرد من العمل":    "التسريح التعسفي",
    "رفت من العمل":    "التسريح التعسفي",
    "طردني صاحب العمل":"التسريح التأديبي التعسفي",
    "فصلني المدير":    "التسريح التأديبي",
    "فقدت عملي":       "إنهاء عقد العمل",
    "مرتبي":           "الأجر",
    "راتبي":           "الأجر",
    "ما دفعولي":       "عدم دفع الأجر",
    "ما خلصوني":       "عدم دفع الأجر",
    "عقدي":            "عقد العمل",
    "عندي عقد":        "عقد العمل",
    "فتح شركة":        "تأسيس الشركة السجل التجاري",
    "نفتح شركة":       "تأسيس الشركة",
    "سرقني":           "جريمة السرقة",
    "ضربني":           "جريمة الضرب والجرح العمد",
    "شتمني":           "جريمة القذف والسب",
    "يطلقها":          "الطلاق",
    "خلع":             "الطلاق بالخلع",
    "مسكن الزوجية":    "السكن الزوجي حق السكن",
    # Roadmap Step 1: Synonym Injection for Cross-Domain Criminal Fraud & more
    "منتجات خطيرة":    "غش سلع ضارة بالصحة المواد 431 432 433 434",
    "إصابة عمل":       "مسؤولية صاحب العمل تعويض حوادث",
    "فوائد التأخير":   "فوائد الديون المدنية التجارية",
    "نصب في التجارة":  "احتيال خداع تدليس قانون العقوبات",
    "نصب التجاري":     "احتيال خداع تدليس قانون العقوبات",
    "نصب":             "احتيال تدليس خداع",
    "خدعني التاجر":    "احتيال تدليس غش تجاري قانون العقوبات",
    "خدعني":           "احتيال تدليس غش",
    "بضاعة مزورة":     "غش تجاري تقليد بضاعة قانون العقوبات",
    "منتج مزور":       "غش تجاري تقليد بضاعة قانون العقوبات",
    "سلعة مزورة":      "غش تجاري تقليد بضاعة قانون العقوبات",
    "بيع منتج مغشوش":  "غش بيع السلع خداع المستهلك قانون العقوبات",
    "لم يسلم السلعة":  "عدم تسليم المبيع احتيال اختلاس قانون العقوبات",
    "عدم تسليم السلعة":"عدم تسليم المبيع احتيال اختلاس قانون العقوبات",
    "ما سلملي":        "عدم تسليم المبيع احتيال اختلاس",
    "تزوير فاتورة":    "تزوير محررات عقوبات",
    "تزوير فواتير":    "تزوير محررات عقوبات",
    "فاتورة مزورة":    "تزوير محررات وثائق قانون العقوبات",
}

def normalize_legal_slang(query: str) -> str:
    normalized = query
    for slang, legal_term in sorted(_LEGAL_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if slang in normalized:
            normalized = normalized.replace(slang, legal_term)
            print(f"[Qwen-Retriever] Normalized '{slang}' → '{legal_term}'")
    return normalized


# ── Public API ─────────────────────────────────────────────────────────────────

def qwen_retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most semantically similar law articles for a query.

    Args:
        query:      User's natural-language question (Arabic preferred).
        top_k:      Number of results to return.
        min_score:  Minimum cosine similarity threshold (0–1).
                    ChromaDB returns *distance* in [0, 2] for cosine space;
                    we convert: similarity = 1 − distance/2   (since vectors
                    are L2-normalised, distance ∈ [0, 2]).

    Returns:
        List of article dicts sorted by descending similarity.
    """
    from qwen_rag.qwen_embedder import embed_query

    collection = _get_collection()

    # Roadmap Step 2: Adaptive top-K for Multi-Article Queries
    adaptive_triggers = ["شروط", "إجراءات", "حقوق", "الفرق بين"]
    if any(t in query for t in adaptive_triggers):
        top_k = max(top_k, 5)
        print(f"[Qwen-Retriever] Adaptive top-K triggered: expanded to {top_k}")

    # Fix E: Apply synonyms (legal normalization)
    query = normalize_legal_slang(query)

    # Embed the query with the instruction prefix
    q_vec = embed_query(query).tolist()[0]

    # Query ChromaDB — fetch more than needed so we can apply min_score filter
    n_results = min(top_k * 3, collection.count())
    response = collection.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    ids        = response["ids"][0]
    documents  = response["documents"][0]
    metadatas  = response["metadatas"][0]
    distances  = response["distances"][0]

    for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        # Convert ChromaDB cosine distance → similarity
        # ChromaDB cosine distance = 1 − cosine_sim  (range 0–2 for unit vecs)
        similarity = round(1.0 - dist, 4)

        if similarity < min_score:
            continue

        results.append({
            "id":                       doc_id,
            "law_name":                 meta.get("law_name", ""),
            "law_domain":               meta.get("law_domain", ""),
            "article_number":           meta.get("article_number", ""),
            "title":                    meta.get("title", ""),
            "text_original":            text,
            "penalties_summary":        meta.get("penalties_summary", ""),
            "legal_conditions_summary": meta.get("legal_conditions_summary", ""),
            "keywords":                 json.loads(meta.get("keywords", "[]")),
            "score":                    similarity,
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    test_queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
        "عقوبة غش المواد الغذائية",
    ]

    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {q}")
        print("=" * 65)
        hits = qwen_retrieve(q, top_k=3)
        if not hits:
            print("  No results found.")
        for i, r in enumerate(hits, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Title : {r['title']}")
            print(f"       Score : {r['score']}")
            print(f"       Text  : {r['text_original'][:150]}…")
