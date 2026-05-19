import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from graph_rag_local.graph_retriever import graph_retrieve
from graph_rag_local.graph_generator import graph_generate

def run_test(model_name, questions):
    print(f"\n[Benchmarking Model: {model_name}]")
    latencies = []
    
    # Primer run (not recorded) to ensure model is in memory
    print(f"  > Priming {model_name}...")
    graph_generate("سلام", [], model=model_name, stream=False)
    
    total_start = time.time()
    for i, q in enumerate(questions):
        print(f"  > Query {i+1}/3...", end="", flush=True)
        # Retrieve context once to keep it constant across models
        articles = graph_retrieve(q, top_k=5)
        
        start = time.time()
        graph_generate(q, articles, model=model_name, stream=False)
        end = time.time()
        
        latency = end - start
        latencies.append(latency)
        print(f" {latency:.2f}s")
        
    total_end = time.time()
    
    return {
        "model": model_name,
        "avg": sum(latencies) / len(latencies),
        "total": total_end - total_start,
        "min": min(latencies),
        "max": max(latencies)
    }

def main():
    questions = [
        "هل النصب في التجارة يُعاقب عليه القانون؟",
        "شخص باعني منتج مزور، هل هذا يعتبر جريمة؟",
        "ما هي عقوبة الغش في بيع السلع؟"
    ]
    
    results = []
    models = ["gemma2:9b", "qwen2:7b"]
    
    for model in models:
        try:
            res = run_test(model, questions)
            results.append(res)
        except Exception as e:
            print(f"Error testing {model}: {e}")

    # Output Comparison Table
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'Avg Latency':<12} | {'Total (3 q)':<12} | {'Min/Max':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['model']:<15} | {r['avg']:>10.2f}s | {r['total']:>10.2f}s | {r['min']:.2f}s / {r['max']:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
