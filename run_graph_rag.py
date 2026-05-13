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

from graph_rag.graph_retriever import graph_retrieve
from gemma_rag.gemma_generator import gemma_generate, OLLAMA_GEMMA_MODEL


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
        f"Using Graph RAG retrieval + {OLLAMA_GEMMA_MODEL} for generation.",
        flush=True,
    )

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"=== Graph RAG Results — Generator: {OLLAMA_GEMMA_MODEL} ===\n\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing: {q}", flush=True)
            start = time.time()

            try:
                chunks = graph_retrieve(q, top_k=5)
                answer = gemma_generate(q, chunks)
            except Exception as e:
                answer = f"Error during processing: {e}"
                chunks = []

            elapsed = time.time() - start

            out.write(f"[User]: {q}\n")
            out.write(f"ANSWER:\n{answer}\n")
            out.write(f"(Time taken: {elapsed:.2f} seconds)\n")
            out.write("SOURCES:\n")
            for idx, chunk in enumerate(chunks, 1):
                law_name   = chunk.get("law_name",       "قانون غير معروف")
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
