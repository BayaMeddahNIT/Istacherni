import time
import sys
from pathlib import Path

def test_import(module_name):
    start = time.time()
    print(f"Importing {module_name}...", end="", flush=True)
    try:
        __import__(module_name)
        print(f" OK ({time.time() - start:.2f}s)")
    except Exception as e:
        print(f" FAIL: {e}")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

test_import("pathlib")
test_import("chromadb")
test_import("sentence_transformers")
test_import("backend.rag.retrieval.local_retriever")
test_import("bm25_rag.bm25_retriever")
test_import("hybrid_rag.hybrid_retriever")
test_import("hybrid_rag.reranker_minilm")
test_import("gemma_rag.gemma_generator")

print("\nAll imports finished.")
