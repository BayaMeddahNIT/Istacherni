import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from agentic_rag.agent_knowledge_base import KB

def debug_search(query):
    print(f"\n🔍 Searching for: '{query}'")
    results = KB.search_articles(query, top_k=5)
    print(f"✅ Found {len(results)} results.")
    for i, res in enumerate(results):
        print(f"  [{i+1}] {res.get('law_name')} - Art {res.get('article_number')}: {res.get('title')}")
        # print(f"      Text sample: {res.get('text_original')[:100]}...")

def list_domains():
    print("\n📋 Available Domains:")
    domains = KB.available_domains()
    for d in domains:
        print(f"  - {d}")

if __name__ == "__main__":
    print("--- KB Debugging ---")
    list_domains()
    debug_search("سرقة")
    debug_search("عقوبة السرقة")
    debug_search("القتل")
