import pickle, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import Counter

corpus_file = Path('graph_rag/cache/graph_corpus.pkl')
if corpus_file.exists():
    with open(corpus_file, 'rb') as f:
        corpus = pickle.load(f)
    law_counts = Counter(a['law_name'] for a in corpus)
    print('Graph corpus law distribution:')
    for law, cnt in law_counts.most_common(15):
        print(f'  {cnt:4d}  {law}')
    print(f'\nTotal articles in graph: {len(corpus)}')
    print('\nSample article_number values:')
    for a in corpus[:5]:
        print(f'  law={a["law_name"]}  art={repr(a["article_number"])}  id={a["id"]}')
    # Check for any multi-article entries
    multi = [a for a in corpus if '-' in str(a['article_number'])]
    print(f'\nMulti-article entries (hyphen in article_number): {len(multi)}')
    for a in multi[:3]:
        print(f'  {a["law_name"]} - المادة {a["article_number"]}')
else:
    print('No corpus cache found at', corpus_file.absolute())
