"""
smoke_test_domain.py
--------------------
Quick retrieval-only smoke test for the two most critical failure modes:
  F4: 'ما هي مسؤولية الشركاء؟'  → should retrieve Commercial Arts 564/568, NOT Civil 419/435
  F3: 'ما هي التزامات المشتري؟'  → should retrieve Arts 364/371/389, NOT Art 754 (contractor)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from graph_rag_local.graph_retriever import graph_retrieve

TESTS = [
    {
        "name": "F4 — Domain Mismatch (Civil vs Commercial)",
        "query": "ما هي مسؤولية الشركاء؟",
        "want_commercial": True,
        "want_articles": ["564", "568", "578"],
        "bad_articles": ["419", "435"],
    },
    {
        "name": "F3 — Scope Drift (Buyer vs Contractor obligations)",
        "query": "ما هي التزامات المشتري؟",
        "want_commercial": False,
        "want_articles": ["364", "371", "389", "361"],
        "bad_articles": ["754"],
    },
]

for test in TESTS:
    print(f"\n{'='*60}")
    print(f"TEST: {test['name']}")
    print(f"Query: {test['query']}")
    chunks = graph_retrieve(test["query"], top_k=7)
    print(f"\nRetrieved {len(chunks)} chunks:")
    hits, misses = 0, 0
    for i, c in enumerate(chunks, 1):
        art_num = str(c.get("article_number", ""))
        domain  = c.get("law_domain", "")
        law     = c.get("law_name", "")
        score   = c.get("graph_score", 0)
        is_good = art_num in test["want_articles"]
        is_bad  = art_num in test["bad_articles"]
        tag = " ✅ WANT" if is_good else (" ❌ BAD" if is_bad else "")
        if is_good: hits += 1
        if is_bad:  misses += 1
        print(f"  [{i}] {law} - م.{art_num}  score={score:.3f}  [{domain}]{tag}")
    print(f"\n  → Wanted articles found: {hits}/{len(test['want_articles'])}")
    print(f"  → Bad articles avoided:  {'YES' if misses == 0 else f'NO ({misses} bad articles retrieved)'}")
