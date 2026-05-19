import json
import sys
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('evaluation_cache.json', 'r', encoding='utf-8') as f:
    cache = json.load(f)

print(f"Total items: {len(cache)}")
for i, (k, v) in enumerate(list(cache.items())[:5]):
    print(f"\n[{i+1}] Key: {k}")
    print(f"Value structure: {list(v.keys()) if isinstance(v, dict) else type(v)}")
    if isinstance(v, dict):
        print(f"Accuracy: {v.get('final_accuracy', 'N/A')}")
