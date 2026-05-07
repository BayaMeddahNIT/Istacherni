import sys
import time
from pathlib import Path

# Add the project root to the python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError in Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("Initializing BGE+BM25 test script... Please wait while heavy libraries load.", flush=True)

from hybrid_rag.hybrid_retriever import hybrid_retrieve
from jais_rag.jais_generator import jais_generate

# NOTE: Original Qwen import (commented out as per supervisor request)
# from qwen_rag.qwen_generator import qwen_generate

def main():
    input_file = PROJECT_ROOT / "questions.txt"
    output_file = PROJECT_ROOT / "answers_bge_bm25_final_1.txt"
    
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
        
    from jais_rag.jais_generator import OLLAMA_JAIS_MODEL
    print(f"Loaded {len(questions)} questions. Using {OLLAMA_JAIS_MODEL} for generation.")
    
    with open(output_file, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing: {q}", flush=True)
            
            start_time = time.time()
            
            try:
                # Retrieve chunks using Hybrid RAG (BGE-M3 + BM25)
                chunks = hybrid_retrieve(q, top_k=5)
                
                # Generate answer using the new SILMA model (via our jais_generate wrapper)
                answer = jais_generate(q, chunks)
                
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
