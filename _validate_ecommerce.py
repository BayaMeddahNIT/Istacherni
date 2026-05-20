import json
data = json.load(open('dataset/raw/ecommerce_law_18_05.json', encoding='utf-8'))
print(f'OK: {len(data)} articles loaded')
for a in data:
    print(f'  Art.{a["article_number"]}: {a["title"]}')
