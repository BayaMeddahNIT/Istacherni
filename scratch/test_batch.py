import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_rag.bge_embedder import _ollama_embed

texts = ["مرحبا", "كيف حالك"]
print("Testing batch embedding...")
try:
    res = _ollama_embed(texts)
    print(f"Success! Result type: {type(res)}")
    if isinstance(res, list):
        print(f"Length of result: {len(res)}")
        if len(res) > 0:
            print(f"Type of first element: {type(res[0])}")
except Exception as e:
    print(f"Failed: {e}")
