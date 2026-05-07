from sentence_transformers import CrossEncoder

print("Attempting to load reranker...", flush=True)
try:
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    print("Loaded!", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
