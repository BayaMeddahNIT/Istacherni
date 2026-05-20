import urllib.request
import json
req = urllib.request.Request(
    'http://localhost:11434/api/generate',
    data=json.dumps({
        'model': 'qwen2:7b',
        'prompt': 'What is the difference between SPA and SARL? Output ONLY JSON: {"is_context_sufficient": "yes", "answer": "..."}',
        'stream': False
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req).read()
print(json.loads(res)['response'])
