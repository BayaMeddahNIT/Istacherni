"""
run_graph_rag.py
----------------
Processes every question in questions.txt through the Knowledge Graph RAG
pipeline and saves the answers to answers_graph_rag.txt.

Retrieval : graph_retrieve()   — Knowledge Graph traversal (3-hop)
Generation: gemma_generate()   — gemma2:9b via Ollama  (local, no API key needed)

Usage:
    python run_graph_rag.py
"""

import sys
import time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── UTF-8 output (Windows) ───────────────────────────────────────────────────────
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

print("Initializing Graph RAG test script… Loading heavy libraries.", flush=True)

# ── Retriever selection ──────────────────────────────────────────────────────────
# False → graph_rag.graph_retriever   (pure keyword/graph traversal, fast, no GPU)
# True  → graph_rag_local.graph_retriever  (BGE-M3 hybrid: embeddings + RRF + PPR
#          + domain boost + HyDE averaging — all 8 fixes applied)
USE_LOCAL_RETRIEVER = True

if USE_LOCAL_RETRIEVER:
    from graph_rag_local.graph_retriever import graph_retrieve
    print("Using LOCAL hybrid retriever (BGE-M3 + PPR + domain boost)")
else:
    from graph_rag.graph_retriever import graph_retrieve
    print("Using KEYWORD graph retriever")
from graph_rag_local.graph_generator import graph_generate
OLLAMA_MODEL = "qwen2:7b"


def main():
    input_file  = PROJECT_ROOT / "questions.txt"
    output_file = PROJECT_ROOT / "answers_graph_rag.txt"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ── Extract questions (skip blank lines and section headers) ─────────────────
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "🟦", "🟨", "🟩", "🟪")):
            continue
        questions.append(line)

    print(
        f"Loaded {len(questions)} questions. "
        f"Using Graph RAG retrieval + {OLLAMA_MODEL} for generation.",
        flush=True,
    )

    with open(output_file, "a", encoding="utf-8") as out:
        # ── Resume support: detect how many questions already answered ──────────
        already_done = 0
        if output_file.exists():
            content = output_file.read_text(encoding="utf-8")
            already_done = content.count("[User]:")
            if already_done > 0:
                print(f"Resuming from question {already_done + 1} (skipping {already_done} already done).")
                questions = questions[already_done:]
        else:
            out.write(f"=== Graph RAG Results (V7 — Fixes A+D+E Active) — Generator: {OLLAMA_MODEL} ===\n\n")


        # ── Step 0: Model Warmup (Load models into memory before timing) ───────────
        if questions:
            print("\n🔥 Warming up models (Initial load into RAM)...", flush=True)
            try:
                # Run a dummy query to force BGE-M3, FAISS, Reranker, and Qwen into memory
                warmup_q = "هل القانون الجزائري يعاقب على السرقة؟"
                warmup_chunks = graph_retrieve(warmup_q, top_k=1)
                if warmup_chunks:
                    _ = graph_generate(warmup_q, warmup_chunks, model=OLLAMA_MODEL, stream=False)
                print("✅ Warmup complete. Starting benchmark.\n")
            except Exception as e:
                print(f"⚠️ Warmup failed (non-critical): {e}\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing: {q}", flush=True)
            start = time.time()

            # ── Retrieval (always runs) ───────────────────────────────────────
            chunks = []
            try:
                # Reverting to baseline top_k=7 as requested by the user.
                chunks = graph_retrieve(q, top_k=7)
            except Exception as e:
                print(f"  [Retrieval ERROR] {e}", flush=True)

            # ── Generation (requires Ollama) ──────────────────────────────────
            answer = ""
            try:
                if chunks:
                    answer = graph_generate(q, chunks, model=OLLAMA_MODEL, stream=False)
                else:
                    answer = "Retrieval returned no results."
            except Exception as e:
                answer = f"Error during generation: {e}"

            elapsed = time.time() - start

            out.write(f"[User]: {q}\n")
            out.write(f"ANSWER:\n{answer}\n")
            out.write(f"(Time taken: {elapsed:.2f} seconds)\n")
            out.write("SOURCES:\n")
            for idx, chunk in enumerate(chunks, 1):
                law_name    = chunk.get("law_name",       "قانون غير معروف")
                article_num = chunk.get("article_number", "N/A")
                graph_score = chunk.get("graph_score",    0)
                pagerank    = chunk.get("pagerank",       0)
                out.write(
                    f"[{idx}] {law_name} - المادة {article_num}"
                    f"  (graph_score={graph_score:.3f}, pagerank={pagerank:.6f})\n"
                )
            out.write("\n" + "=" * 50 + "\n\n")
            out.flush()

    print(f"\nDone! {len(questions)} answers saved to {output_file}")


if __name__ == "__main__":
    main()
