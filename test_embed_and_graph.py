import time
from graph_rag_local.embeddings import embed_text
from graph_rag_local.graph_builder import GRAPH_FILE
import networkx as nx
import pickle

text = 'ما عقوبة النصب والاحتيال في القانون الجزائري؟'
print('Warming up BGE-M3...')
# First call loads the model
start = time.time()
emb = embed_text(text, show_progress=False)
print(f'First call (with loading): {time.time() - start:.2f} s')

print('Measuring cached call...')
start = time.time()
emb2 = embed_text(text, show_progress=False)
print(f'Second call (warm): {time.time() - start:.2f} s')

print('\nAnalyzing Domain Distribution...')
if GRAPH_FILE.exists():
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    domains = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "article":
            domain = d.get("law_domain", "Unknown")
            domains[domain] = domains.get(domain, 0) + 1
    total = sum(domains.values())
    for d, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d}: {count} articles ({count/total*100:.1f}%)")
    print(f"Total articles: {total}")
