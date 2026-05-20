"""
smoke_test_ecommerce.py
-----------------------
Validates that the newly indexed القانون 18-05 articles are correctly retrieved
for the benchmark's hardest failure: 'هل البيع عبر الإنترنت قانوني؟' (LA=0.075 before fix).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from graph_rag_local.graph_retriever import graph_retrieve

TESTS = [
    {
        "name": "F1 — Knowledge Gap: Online Sales Legality",
        "query": "هل البيع عبر الإنترنت قانوني؟",
        "want_articles": ["1", "2", "3"],  # GT articles from benchmark
        "want_law": "القانون 18-05",
    },
    {
        "name": "F1b — E-commerce Registration Penalty",
        "query": "هل يجب التسجيل في السجل التجاري للبيع عبر الإنترنت؟",
        "want_articles": ["31", "36", "3"],
        "want_law": "القانون 18-05",
    },
]

for test in TESTS:
    print(f"\n{'='*65}")
    print(f"TEST: {test['name']}")
    print(f"Query: {test['query']}")
    chunks = graph_retrieve(test["query"], top_k=7)
    print(f"\nRetrieved {len(chunks)} chunks:")
    hits = 0
    for i, c in enumerate(chunks, 1):
        art_num = str(c.get("article_number", ""))
        law     = c.get("law_name", "")
        score   = c.get("graph_score", 0)
        domain  = c.get("law_domain", "")
        is_ecom = test["want_law"] in law
        is_want = art_num in test["want_articles"] and is_ecom
        if is_want: hits += 1
        tag = " ✅ TARGET" if is_want else (" 🔵 LAW18-05" if is_ecom else "")
        print(f"  [{i}] {law} - م.{art_num}  score={score:.3f}{tag}")
    print(f"\n  → Target articles found: {hits}/{len(test['want_articles'])}")
    success = hits > 0
    print(f"  → Status: {'✅ PASS — E-commerce law indexed and retrieved' if success else '❌ FAIL — Still no e-commerce articles'}")
