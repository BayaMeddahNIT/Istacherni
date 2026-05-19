"""
Identify exactly which Penal Code article numbers are in the RAGAS GT dataset
but missing from the indexed corpus — the specific gap to fill.
"""
import json, sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# Load GT articles
dataset = json.loads(Path('algerian_law_ragas_dataset_v3.json').read_text(encoding='utf-8'))
gt_penal = Counter()
gt_other = Counter()
for item in dataset:
    for art in item.get('articles', []):
        if 'عقوبات' in art:
            # Extract article number
            import re
            m = re.search(r'المادة\s+(\S+)', art)
            if m:
                gt_penal[m.group(1)] += 1
        else:
            gt_other[art.split(' - المادة')[0].strip()] += 1

print(f'GT dataset: {len(dataset)} questions')
print(f'Unique Penal Code articles in GT: {len(gt_penal)}')
print(f'Unique non-Penal articles in GT: {len(Counter(gt_other.keys()))}')
print()
print('Non-penal law coverage in GT:')
for law, cnt in gt_other.most_common(10):
    print(f'  {cnt:3d}  {law}')

# Load indexed corpus
import pickle
corpus_file = Path('graph_rag/cache/graph_corpus.pkl')
with open(corpus_file, 'rb') as f:
    corpus = pickle.load(f)

indexed_penal = set()
for a in corpus:
    if 'عقوبات' in a.get('law_name', ''):
        indexed_penal.add(str(a['article_number']))

print()
print(f'Indexed Penal Code articles: {len(indexed_penal)}')
print(f'GT Penal Code articles needed: {len(gt_penal)}')
missing = set(gt_penal.keys()) - indexed_penal
print(f'MISSING from index: {len(missing)} articles')
print()
print('Most-needed missing Penal Code articles (by GT frequency):')
missing_by_freq = [(art, gt_penal[art]) for art in missing]
missing_by_freq.sort(key=lambda x: x[1], reverse=True)
for art, cnt in missing_by_freq[:30]:
    print(f'  المادة {art:10s}  (appears in GT {cnt}x)')

print()
# Which GT questions have ZERO indexed sources?
zero_coverage = []
indexed_all = {}
for a in corpus:
    law = a.get('law_name', '')
    art = str(a.get('article_number', ''))
    key = f"{law} - المادة {art}"
    indexed_all[key] = True

for item in dataset:
    arts = item.get('articles', [])
    covered = any(a in indexed_all for a in arts)
    if not covered:
        zero_coverage.append(item['question'][:60])

print(f'GT questions with ZERO indexed coverage: {len(zero_coverage)}/{len(dataset)}')
print('Sample:')
for q in zero_coverage[:5]:
    print(f'  {q}')
