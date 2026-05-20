import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from graph_rag_local.graph_retriever import graph_retrieve, CACHE_DIR

def clear_query_cache():
    print("[Benchmark] Clearing persistent disk cache...")
    for p in CACHE_DIR.glob("graph_retrieval_cache.db*"):
        try:
            os.remove(p)
            print(f"[Benchmark] Deleted: {p.name}")
        except Exception as e:
            print(f"[Benchmark] Failed to delete {p.name}: {e}")

def run_benchmark(query, label):
    print(f"\n>>> Running: {label}")
    start = time.time()
    
    # Run retrieval only
    articles = graph_retrieve(query, top_k=10)
    
    end = time.time()
    latency = end - start
    print(f"<<< Completed in {latency:.3f} seconds ({len(articles)} articles found)")
    return latency

def main():
    query = "ما هي عقوبة التزوير في المحررات الرسمية؟"
    
    print("=== ISTACHERNI PERFORMANCE LOG ===")
    clear_query_cache()
    
    # 1. COLD START
    cold_latency = run_benchmark(query, "COLD START (Full Retrieval + Rerank + PPR)")
    
    # 2. WARM START
    warm_latency = run_benchmark(query, "WARM START (Disk Cache Hit)")
    
    if warm_latency > 0:
        improvement = (cold_latency / warm_latency)
        gain_pct = ((cold_latency - warm_latency) / cold_latency) * 100
        print(f"\n{'='*50}")
        print(f"CACHE SPEEDUP: {improvement:.1f}x faster")
        print(f"LATENCY SAVINGS: {gain_pct:.1f}%")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
