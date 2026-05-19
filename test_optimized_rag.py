import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent))

from graph_rag_local.graph_retriever import graph_retrieve
from graph_rag_local.graph_generator import graph_generate

def main():
    query = "كيف أتعامل مع ديون الشركة إذا كنت شريكاً في شركة ذات مسؤولية محدودة؟"
    print(f"--- TESTING OPTIMIZED RAG ---")
    print(f"Query: {query}\n")
    
    print("[1/2] Starting Parallel Retrieval...")
    # top_k=12 to test the context tiering (top-5 vs others)
    articles = graph_retrieve(query, top_k=12)
    print(f"Retrieved {len(articles)} articles.\n")
    
    print("[2/2] Starting Streamed Generation...")
    # This will print to stdout as it generates
    answer = graph_generate(query, articles, stream=True)
    
    print(f"\n\n--- TEST COMPLETE ---")
    print(f"Total characters: {len(answer)}")

if __name__ == "__main__":
    main()
