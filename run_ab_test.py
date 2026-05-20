
import json
import os
import time
import sys
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.getcwd())

from graph_rag_local.graph_retriever import graph_retrieve
from graph_rag_local.graph_generator import graph_generate

def load_failures(cache_path, threshold=0.3):
    """
    Loads failed queries from the evaluation cache.
    Looks for "||Graph RAG||bge" entries with final_accuracy < threshold.
    """
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    
    failures = []
    for key, metrics in cache.items():
        if "||Graph RAG||bge" in key:
            # Note: The user said Legal Accuracy < 0.3. 
            # In the cache, 'final_accuracy' seems to be the key for the weighted score.
            accuracy = metrics.get('final_accuracy', 0)
            if accuracy < threshold:
                query = key.split('||')[0]
                failures.append({
                    "query": query,
                    "old_metrics": metrics
                })
    return failures

def run_ab_test(failures, limit=None):
    progress_path = 'evaluation/ab_test_in_progress.json'
    results = []
    processed_queries = set()
    
    if os.path.exists(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_queries = {item['query'] for item in results}
        print(f"Resuming A/B test. Already processed {len(processed_queries)} queries.")

    if limit:
        # Only take 'limit' new failures
        remaining = [f for f in failures if f['query'] not in processed_queries]
        failures_to_run = remaining[:limit]
    else:
        failures_to_run = [f for f in failures if f['query'] not in processed_queries]
        
    print(f"Starting A/B test on {len(failures_to_run)} new failed queries...")
    
    for i, item in enumerate(failures_to_run):
        query = item['query']
        print(f"\n[{len(results)+1}/{len(failures)}] Query: {query}")
        
        start_time = time.time()
        # Retrieve articles
        articles = graph_retrieve(query, top_k=10) # Using top_k=10 to allow for tiering
        
        # Generate answer with new Top-3 Tiering
        # We pass stream=False here for easier handling in the script
        new_answer = graph_generate(query, articles, stream=False, model="qwen2:7b")
        elapsed = time.time() - start_time
        
        print(f"Done in {elapsed:.2f}s")
        
        results.append({
            "query": query,
            "old_metrics": item['old_metrics'],
            "new_answer": new_answer,
            "elapsed_s": elapsed,
            "sources": [f"{a['law_name']} - {a['article_number']}" for a in articles]
        })
        
        # Save intermediate results
        with open('evaluation/ab_test_in_progress.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results

if __name__ == "__main__":
    cache_file = 'evaluation_cache.json'
    if not os.path.exists(cache_file):
        print(f"Error: {cache_file} not found.")
        sys.exit(1)
        
    failures = load_failures(cache_file)
    print(f"Found {len(failures)} queries with Legal Accuracy < 0.3.")
    
    # Check for --full flag
    limit = 3 # Default to small batch for verification
    if "--full" in sys.argv:
        limit = None
    
    results = run_ab_test(failures, limit=limit)
    
    output_file = f"evaluation/ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('evaluation', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\nA/B Test complete. Results saved to {output_file}")
