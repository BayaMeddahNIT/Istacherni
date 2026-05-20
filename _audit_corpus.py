"""
Audit the raw dataset directory to count articles by law_name
and identify the actual Penal Code coverage gap.
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

RAW_DIR = Path('dataset/raw')
law_counter = Counter()
file_summary = defaultdict(list)  # file -> [law_names]
article_total = 0
penal_articles = []

def extract_articles(data):
    if isinstance(data, list):
        while len(data) == 1 and isinstance(data[0], list):
            data = data[0]
        return [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        return [data]
    return []

for path in sorted(RAW_DIR.glob('*.json')):
    if path.name.startswith(('add_', 'test', 'algerian')):
        continue
    try:
        content = path.read_text(encoding='utf-8-sig')
        data = json.loads(content)
        articles = extract_articles(data)
        for a in articles:
            if not isinstance(a, dict):
                continue
            art_num = a.get('article_number', '')
            if art_num is None or (isinstance(art_num, list) and len(art_num) == 0):
                continue
            law = a.get('law_name', 'UNKNOWN')
            law_counter[law] += 1
            article_total += 1
            file_summary[path.name].append(law)
            if 'عقوبات' in law or 'Penal' in path.name.lower() or 'penal' in path.name.lower() or 'jinayat' in path.name.lower():
                penal_articles.append({'file': path.name, 'law': law, 'art': str(art_num)[:30]})
    except Exception as e:
        print(f'SKIP {path.name}: {e}')

print(f'Total articles found: {article_total}')
print()
print('Law distribution across all JSON files:')
for law, cnt in law_counter.most_common(20):
    print(f'  {cnt:5d}  {law}')

print()
print('Penal Code articles found:')
print(f'  Total penal articles: {len(penal_articles)}')
by_file = Counter(a["file"] for a in penal_articles)
for f, cnt in by_file.most_common():
    print(f'  {cnt:4d}  {f}')

print()
print('Per-file breakdown:')
for fname, laws in sorted(file_summary.items()):
    law_dist = Counter(laws)
    print(f'  {fname}: {dict(law_dist.most_common(3))}  ({len(laws)} articles)')
