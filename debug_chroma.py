# debug_chroma.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import chromadb

CHROMA_DIR = Path("dense_rag/chroma_db")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

print("=== Collections ===")
collections = client.list_collections()
print(f"Found {len(collections)} collection(s):")
for c in collections:
    print(f"  - {c.name}")

if collections:
    col = client.get_collection(collections[0].name)
    count = col.count()
    print(f"\nCollection '{col.name}' has {count} documents")
    
    if count > 0:
        # Peek at first 3 docs
        sample = col.peek(3)
        print("\nSample documents:")
        for i, (doc_id, meta) in enumerate(zip(sample["ids"], sample["metadatas"])):
            print(f"  [{i}] id={doc_id} | meta keys={list(meta.keys())}")
            print(f"       id_in_meta={meta.get('id', 'MISSING ⚠️')}")
    else:
        print("\n⚠️  Collection is EMPTY — index was not built properly!")
else:
    print("\n⚠️  No collections found — bge_indexer.py was never run successfully!")