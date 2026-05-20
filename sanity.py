import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from graph_rag_local.graph_retriever import graph_retrieve
import json

q = "ما هي عقوبة السرقة بالإكراه؟"
articles = graph_retrieve(q, top_k=5)

print(f"Retrieved {len(articles)} articles.")
for i, art in enumerate(articles, 1):
    print(f"\n--- Article {i} ---")
    print(f"ID: {art['id']}")
    print(f"Article Number: {art['article_number']}")
    print(f"Title: {art['title']}")
    print(f"Text Original (first 100 chars): {art['text_original'][:100]}...")
    print(f"Penalties Summary: {art.get('penalties_summary')}")
    print(f"Legal Conditions Summary: {art.get('legal_conditions_summary')}")
