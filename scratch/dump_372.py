import json
import sys
from pathlib import Path

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

path = Path("dataset/raw/Penal Code/Jinayat.json")
with open(path, encoding="utf-8") as f:
    data = json.load(f)
    for art in data:
        if art.get("article_number") == 372:
            print(json.dumps(art, ensure_ascii=False, indent=2))
            break

