import json
import sys
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('evaluation_cache.json', 'r', encoding='utf-8') as f:
    cache = json.load(f)

print(f"Total items: {len(cache)}")
for i, (k, v) in enumerate(list(cache.items())):
    if "||Graph RAG||bge" in k:
        print(f"[{i+1}] {k}")
        print(f"    - Accuracy: {v.get('final_accuracy', 'N/A')}")
