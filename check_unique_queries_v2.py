import json
import sys
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('evaluation_cache.json', 'r', encoding='utf-8') as f:
    cache = json.load(f)

queries = set()
for k in cache.keys():
    parts = k.split('||')
    if len(parts) >= 2:
        # For 'ragas||question||answer', question is at index 1
        # For 'question||model||embedding', question is at index 0
        if parts[0] in ['ragas', 'emb', 'corr']:
            queries.add(parts[1])
        else:
            queries.add(parts[0])
    else:
        queries.add(k)

print(f"Total items in cache: {len(cache)}")
print(f"Unique queries in cache: {len(queries)}")

# Print some unique queries
print("\nSample queries:")
for q in list(queries)[:10]:
    print(f"- {q}")
