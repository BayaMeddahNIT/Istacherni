import sys
import time
from pathlib import Path

# Add the project root to the python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError in Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("Initializing Gemma+BM25 test script... Please wait while heavy libraries load.", flush=True)

from backend.rag.retrieval.qwen_retriever import retrieve as qwen_retrieve
from bm25_rag.bm25_retriever import bm25_retrieve
from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion
from hybrid_rag.reranker_qwen import rerank_candidates
from gemma_rag.gemma_generator import gemma_generate

def hybrid_retrieve_qwen_bm25(query: str, top_k: int = 5) -> list[dict]:
    fetch_k = 20
    
    # 1. BM25 Retrieval
    bm25_results = bm25_retrieve(query, top_k=fetch_k)
    
    # 2. Qwen3-Embedding Retrieval (Keeping Qwen for embeddings as requested)
    qwen_raw = qwen_retrieve(query, top_k=fetch_k)
    
    # Adapt Qwen3 results to match bm25's flat structure for easy fusion & generation
    qwen_results = []
    for r in qwen_raw:
        meta = r.get("metadata", {})
        adapted = {
            "id": r["id"],
            "score": r["score"],
            "retrieval_method": "dense",
            "text_original": r.get("text", meta.get("text_original", "")), # fallback
            **meta
        }
        qwen_results.append(adapted)
        
    # 3. Fuse results using Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(bm25_results, qwen_results)
    
    # 4. Rerank the fused results
    reranked = rerank_candidates(query, fused, top_k=top_k)
    return reranked

def main():
    input_file = PROJECT_ROOT / "questions.txt"
    output_file = PROJECT_ROOT / "answers_qwen_bm25_final.txt"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Extract questions
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip headers like "1. Penal Law...", color emojis, etc.
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "🟦", "🟨", "🟩", "🟪")):
            continue
        questions.append(line)
        
    print(f"Loaded {len(questions)} questions from {input_file.name}. Using Gemma 2 for generation.")
    
    with open(output_file, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing: {q}")
            
            start_time = time.time()
            
            try:
                # Retrieve chunks using Hybrid RAG (Qwen3 + BM25)
                chunks = hybrid_retrieve_qwen_bm25(q, top_k=5)
                
                # Generate answer using local Gemma 2
                answer = gemma_generate(q, chunks)
                
            except Exception as e:
                answer = f"Error during processing: {e}"
                chunks = []
                
            elapsed = time.time() - start_time
            
            # Format output exactly as requested
            out.write(f"[User]: {q}\n")
            out.write(f"ANSWER:\n{answer}\n")
            out.write(f"(Time taken: {elapsed:.2f} seconds)\n")
            out.write("SOURCES:\n")
            for idx, chunk in enumerate(chunks, 1):
                law_name = chunk.get("law_name", "قانون غير معروف")
                article_num = chunk.get("article_number", "N/A")
                out.write(f"[{idx}] {law_name} - المادة {article_num}\n")
            
            out.write("\n" + "="*50 + "\n\n") # Separation for the next query
            out.flush() # Ensure it's saved continuously in case of crash
            
    print(f"\nDone! All {len(questions)} answers have been saved to {output_file}")

if __name__ == "__main__":
    main()
