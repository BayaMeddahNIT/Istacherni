import time
import os
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from graph_rag_local.graph_retriever import graph_retrieve, CACHE_DIR
from graph_rag_local.graph_generator import graph_generate

def clear_query_cache():
    """Removes the persistent retrieval cache to force a cold start."""
    cache_path = CACHE_DIR / "graph_retrieval_cache.db"
    # Shelve creates multiple files (e.g. .dat, .bak, .dir)
    for p in CACHE_DIR.glob("graph_retrieval_cache.db*"):
        try:
            os.remove(p)
            print(f"Cleared cache file: {p.name}")
        except:
            pass

def benchmark_query(query: str, label: str):
    print(f"\n--- {label} ---")
    start_time = time.time()
    
    # Measure Retrieval
    ret_start = time.time()
    articles = graph_retrieve(query, top_k=10)
    ret_end = time.time()
    
    # Measure Generation (First 50 chars to avoid waiting for full streaming during benchmark)
    # We use stream=False here to get the full response time accurately
    gen_start = time.time()
    answer = graph_generate(query, articles, stream=False)
    gen_end = time.time()
    
    total_time = time.time() - start_time
    
    print(f"Retrieval Time: {ret_end - ret_start:.2f}s")
    print(f"Generation Time: {gen_end - gen_start:.2f}s")
    print(f"Total Latency: {total_time:.2f}s")
    return total_time

def main():
    query = "ما هي مسؤولية الشريك عن ديون الشركة ذات المسؤولية المحدودة؟"
    
    print("Pre-clearing cache for Benchmark...")
    clear_query_cache()
    
    # 1. Cold Start
    cold_time = benchmark_query(query, "COLD START (Cache Miss)")
    
    # 2. Warm Start
    warm_time = benchmark_query(query, "WARM START (Cache Hit)")
    
    improvement = ((cold_time - warm_time) / cold_time) * 100
    print(f"\n{'='*40}")
    print(f"PERFORMANCE GAIN: {improvement:.1f}% reduction in latency")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
