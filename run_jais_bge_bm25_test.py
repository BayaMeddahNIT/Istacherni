import sys
import time
import traceback
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safer encoding setup for Windows
try:
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass # Some environments don't support reconfigure

print("=" * 60, flush=True)
print("  DEBUG: Jais + BGE-M3 + BM25 Test Runner", flush=True)
print("=" * 60, flush=True)

# ── Detailed Dependency Tracing ────────────────────────────────────────────────
print("[TRACE] 1. Importing chromadb...", flush=True)
import chromadb
print("[TRACE] 2. Importing torch...", flush=True)
import torch
print("[TRACE] 3. Importing hybrid_retriever sub-components...", flush=True)

print("[TRACE] 3.1. Importing bm25_retrieve...", flush=True)
from bm25_rag.bm25_retriever import bm25_retrieve
print("[TRACE] 3.2. Importing dense_retrieve...", flush=True)
from dense_rag.bge_retriever import dense_retrieve

print("[TRACE] 3.3. Importing hybrid_retrieve...", flush=True)
try:
    from hybrid_rag.hybrid_retriever import hybrid_retrieve
    print("[STEP 1] OK: hybrid_retrieve loaded.", flush=True)
except Exception as _e:
    print(f"[STEP 1] FAILED: {_e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("[TRACE] 4. Importing jais_generator...", flush=True)
try:
    from jais_rag.jais_generator import jais_generate, check_jais_health, OLLAMA_JAIS_MODEL
    print("[STEP 2] OK: jais_generate loaded.", flush=True)
except Exception as _e:
    print(f"[STEP 2] FAILED: {_e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# NOTE: To switch back to Qwen uncomment the line below (and replace jais_generate calls):
# from qwen_rag.qwen_generator import qwen_generate

# ── Jais Local check ───────────────────────────────────────────────────────────
print(f"[INFO] Jais local model: {OLLAMA_JAIS_MODEL}", flush=True)

# ── Optional: quick health check ──────────────────────────────────────────────
print("[STEP 3] Running Jais local health check ...", flush=True)
ok = check_jais_health()
if not ok:
    print("[ERROR] Jais model not found in Ollama. Please run: ollama pull jwnder/jais-adaptive:7b", flush=True)
    sys.exit(1)
else:
    print("[STEP 3] OK: Jais ready in Ollama.", flush=True)

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    input_file  = PROJECT_ROOT / "questions.txt"
    output_file = PROJECT_ROOT / "answers_jais_bge_bm25.txt"

    print(f"\n[INFO] Questions file : {input_file}", flush=True)
    print(f"[INFO] Output file    : {output_file}", flush=True)

    if not input_file.exists():
        print(f"\n[ERROR] {input_file} not found. Make sure questions.txt is in the project root.", flush=True)
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract questions — same filtering logic as run_bge_bm25_test.py
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "🟦", "🟨", "🟩", "🟪")):
            continue
        questions.append(line)

    if not questions:
        print("[ERROR] No questions parsed from questions.txt. File may be empty or all lines were filtered.", flush=True)
        return

    print(f"[INFO] Loaded {len(questions)} questions.", flush=True)
    print("-" * 60, flush=True)

    with open(output_file, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q}", flush=True)
            start_time = time.time()

            try:
                # Step A: Hybrid retrieval (BGE-M3 + BM25)
                chunks = hybrid_retrieve(q, top_k=5)
                print(f"  └─ Retrieved {len(chunks)} chunks.", flush=True)

                # Step B: Jais-30B generation via HuggingFace API
                answer = jais_generate(q, chunks)
                print(f"  └─ Answer generated ({len(answer)} chars).", flush=True)

            except Exception as e:
                print(f"  └─ ERROR: {e}", flush=True)
                traceback.print_exc()
                answer = f"Error during processing: {e}"
                chunks = []

            elapsed = time.time() - start_time

            out.write(f"[User]: {q}\n")
            out.write(f"ANSWER:\n{answer}\n")
            out.write(f"(Time taken: {elapsed:.2f} seconds)\n")
            out.write("SOURCES:\n")
            for idx, chunk in enumerate(chunks, 1):
                law_name    = chunk.get("law_name",       "قانون غير معروف")
                article_num = chunk.get("article_number", "N/A")
                out.write(f"[{idx}] {law_name} - المادة {article_num}\n")
            out.write("\n" + "=" * 50 + "\n\n")
            out.flush()

    print(f"\n[DONE] All {len(questions)} answers saved to: {output_file}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.", flush=True)
    except Exception as _top_e:
        print(f"\n[FATAL] Unexpected error: {_top_e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

