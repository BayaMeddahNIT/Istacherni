import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_rag.hybrid_retriever import hybrid_retrieve
from qwen_rag.qwen_generator import qwen_generate

query = "ما هي عقوبة النصب في المادة 372 من قانون العقوبات؟"
# print(f"Testing End-to-End for: {query}")

chunks = hybrid_retrieve(query, top_k=5)
answer = qwen_generate(query, chunks)

with open("scratch/full_gen_verify.txt", "w", encoding="utf-8") as out:
    out.write(f"USER: {query}\n\n")
    out.write(f"ANSWER:\n{answer}\n\n")
    out.write("SOURCES:\n")
    for i, c in enumerate(chunks, 1):
        out.write(f"[{i}] {c.get('law_name')} - {c.get('article_number')}\n")

print("Done. Check scratch/full_gen_verify.txt")
