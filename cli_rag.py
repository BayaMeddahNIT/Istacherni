
import sys
import os
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("--- Istacherni Terminal RAG ---")
print("Initializing components...")

try:
    from bm25_rag.hybrid_bge_qwen_retriever import hybrid_retrieve
    from bm25_rag.hybrid_bge_qwen_generator import qwen_generate
    print("✓ Components loaded.")
except ImportError as e:
    print(f"Error: Could not import RAG components. {e}")
    sys.exit(1)

def main():
    print("\nWelcome to the Algerian Law Terminal Assistant!")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        question = input("\n[User]: ").strip()
        
        if not question:
            continue
        if question.lower() in ["exit", "quit", "exit()"]:
            print("Goodbye!")
            break
            
        print("\n[Thinking] Searching for relevant law articles...")
        start_time = time.time()
        
        try:
            # 1. Retrieval
            chunks = hybrid_retrieve(question, top_k=5)
            if not chunks:
                print("[!] No relevant articles found.")
                continue
                
            print(f"[Thinking] Found {len(chunks)} articles. Generating answer with Qwen2.5...")
            
            # 2. Generation
            answer = qwen_generate(question, chunks)
            
            elapsed = time.time() - start_time
            
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(answer)
            print("="*60)
            print(f"\n(Time taken: {elapsed:.2f} seconds)")
            
            print("\nSOURCES:")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] {c.get('law_name')} - المادة {c.get('article_number')}")

        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            print("Make sure 'ollama serve' is running and you have the model pulled.")

if __name__ == "__main__":
    main()
