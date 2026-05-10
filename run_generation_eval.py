import sys
import time
import json
from pathlib import Path

# Add the project root to the python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError in Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("Initializing End-to-End Generation Evaluation... Please wait while models load.", flush=True)

from hybrid_rag.hybrid_retriever import hybrid_retrieve
from gemma_rag.gemma_generator import gemma_generate, OLLAMA_GEMMA_MODEL

def main():
    input_file = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
    output_file = PROJECT_ROOT / "answers_gemma2_finetuned.txt"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.") 
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} questions from {input_file.name}. Using {OLLAMA_GEMMA_MODEL} for generation.")
    
    # Check if there's already an output file to resume from
    processed_count = 0
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            processed_count = content.count("ANSWER:\n")
        print(f"Found existing output file. Resuming from question {processed_count + 1}...")
    
    with open(output_file, "a" if processed_count > 0 else "w", encoding="utf-8") as out:
        for i in range(processed_count, len(dataset)):
            q = dataset[i].get("question", "")
            if not q:
                continue
                
            print(f"[{i+1}/{len(dataset)}] Processing: {q}", flush=True)
            
            start_time = time.time()
            
            try:
                # Retrieve chunks using fine-tuned Hybrid RAG (BGE-M3 + BM25)
                chunks = hybrid_retrieve(q, top_k=5, bm25_weight=0.3, dense_weight=0.7)
                
                # Generate answer using Gemma 2
                answer = gemma_generate(q, chunks)
                
            except Exception as e:
                answer = f"Error during processing: {e}"
                chunks = []
                
            elapsed = time.time() - start_time
            
            # Format output exactly as requested by evaluate_custom_judge.py
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
            
    print(f"\nDone! All {len(dataset)} answers have been saved to {output_file.name}")

if __name__ == "__main__":
    main()
