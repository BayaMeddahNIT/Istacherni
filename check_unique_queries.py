import json
import sys
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('evaluation_cache.json', 'r', encoding='utf-8') as f:
    cache = json.load(f)

queries = set()
for k in cache.keys():
    # Try to extract query before ||
    if '||' in k:
        q = k.split('||')[0]
        queries.add(q)
    else:
        queries.add(k)

print(f"Total items in cache: {len(cache)}")
print(f"Unique queries in cache: {len(queries)}")

# Print 10 unique queries
print("\nSample queries:")
for q in list(queries)[:10]:
    print(f"- {q}")
