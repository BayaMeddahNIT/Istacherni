"""Quick diagnostic — run from project root."""
import json
from pathlib import Path

data_dir = Path("dataset/raw")
print(f"Exists: {data_dir.exists()}")
print(f"Absolute: {data_dir.absolute()}\n")

files = list(data_dir.rglob("*.jsonl")) + list(data_dir.rglob("*.json"))
print(f"Files found: {len(files)}")
for f in files:
    print(f"  {f.name}  ({f.suffix})  size={f.stat().st_size} bytes")

print()
for f in files[:3]:
    print(f"--- {f.name} ---")
    text = f.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    print(f"  Total lines: {len(lines)}")
    # Try parsing first line
    if lines:
        try:
            obj = json.loads(lines[0])
            print(f"  First line keys: {list(obj.keys())[:8]}")
        except Exception as e:
            print(f"  First line parse error: {e}")
            print(f"  First 100 chars: {lines[0][:100]}")

