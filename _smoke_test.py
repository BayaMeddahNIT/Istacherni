import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from graph_rag_local.graph_retriever import graph_retrieve

q = 'هل النصب في التجارة يعاقب عليه القانون؟'
results = graph_retrieve(q, top_k=5)

print('Query:', q)
print()
print('Top 5 retrieved:')
for i, r in enumerate(results, 1):
    print(f'  [{i}] {r["law_name"]} - المادة {r["article_number"]}  (score={r["graph_score"]})')

print()
laws  = [r['law_name'] for r in results]
penal = sum(1 for l in laws if 'عقوبات' in l)
civil = sum(1 for l in laws if 'مدني' in l)
print(f'Penal Code articles: {penal}/5')
print(f'Civil Code articles: {civil}/5')
print()
if penal >= 2:
    print('Fix 1 WORKING — domain inversion resolved')
elif civil == 5:
    print('Fix 1 NOT WORKING — still returning all Civil Code')
else:
    print('Partial improvement — check domain classification output above')
