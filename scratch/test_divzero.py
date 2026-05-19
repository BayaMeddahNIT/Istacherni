import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import traceback
from hybrid_rag.hybrid_retriever import hybrid_retrieve

try:
    print("Testing hybrid_retrieve...")
    results = hybrid_retrieve("ماذا أفعل إذا خانني شريكي؟", top_k=5)
    print("Success, got", len(results), "results")
except Exception as e:
    print("Caught exception:")
    traceback.print_exc()
