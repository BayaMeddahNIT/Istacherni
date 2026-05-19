"""
Verify the multi-article source string problem: check which Commercial Code
articles in the graph have merged article numbers that can never match GT.
"""
import pickle, json, re, sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# Load corpus
with open('graph_rag/cache/graph_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

# Find multi-article entries
multi = [(a['law_name'], a['article_number'], a['id']) for a in corpus if '-' in str(a['article_number'])]
print(f'Multi-article entries in indexed corpus: {len(multi)}')
for law, art, aid in multi[:20]:
    print(f'  [{aid}] {law} - المادة {art}')

# Load GT and see which of these numbers appear there individually
dataset = json.loads(Path('algerian_law_ragas_dataset_v3.json').read_text(encoding='utf-8'))
gt_set = set()
for item in dataset:
    for a in item.get('articles', []):
        gt_set.add(a)

print()
print('Checking if any sub-numbers of merged entries appear in GT:')
total_sub_matches = 0
for law, art_str, aid in multi:
    sub_nums = [s.strip() for s in str(art_str).split('-') if s.strip()]
    for sub in sub_nums:
        candidate = f'{law} - المادة {sub}'
        if candidate in gt_set:
            print(f'  HIT: {candidate}  (from merged {aid})')
            total_sub_matches += 1
print(f'Total sub-number GT matches that are being missed: {total_sub_matches}')
