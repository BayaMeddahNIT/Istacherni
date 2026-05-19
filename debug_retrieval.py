import sys
import io
import pickle
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from graph_rag_local.graph_retriever import graph_retrieve, _G

# Load graph manually to check node data
CACHE_DIR = Path("graph_rag_local/cache")
GRAPH_FILE = CACHE_DIR / "law_graph_local.pkl"

with open(GRAPH_FILE, "rb") as f:
    G = pickle.load(f)

q = "ما هي عقوبة السرقة بالإكراه؟"
articles = graph_retrieve(q, top_k=5)
for i, a in enumerate(articles, 1):
    art_id = a.get('id')
    node_data = G.nodes[art_id]
    print(f"Article {i}: ID={art_id}, Law={node_data.get('law_name')}, ArtNum={node_data.get('article_number')}")
    print(f"Text Original (first 100): '{node_data.get('text_original', '')[:100]}'")
    print(f"Search Block (first 100): '{node_data.get('search_block', '')[:100]}'")
    print("-" * 20)