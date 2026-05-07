import os
import json
from pathlib import Path

raw_dir = Path("dataset/raw")
for f in raw_dir.rglob("*.json"):
    try:
        with open(f, encoding="utf-8") as j:
            data = json.load(j)
            if not isinstance(data, list): continue
            for art in data:
                # Check for "372" in article_number or text
                num = str(art.get("article_number", ""))
                if "372" in num:
                    print(f"File: {f.relative_to(raw_dir)}")
                    print(f"Law: {art.get('law_name')}")
                    print(f"Article: {art.get('article_number')}")
                    print(f"Title: {art.get('title')}")
                    print(f"Text Snippet: {str(art.get('text'))[:100]}")
                    print("-" * 20)
    except Exception as e:
        pass

