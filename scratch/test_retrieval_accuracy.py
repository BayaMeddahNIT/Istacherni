import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_rag.hybrid_retriever import hybrid_retrieve

query = "ما هي عقوبة النصب في المادة 372 من قانون العقوبات؟"
with open("scratch/retrieval_verify_results.txt", "w", encoding="utf-8") as out:
    out.write(f"Testing Query: {query}\n")
    out.write("-" * 50 + "\n")

    results = hybrid_retrieve(query, top_k=5)

    for i, res in enumerate(results, 1):
        law = res.get("law_name", "Unknown")
        art = res.get("article_number", "N/A")
        score = res.get("rerank_score", "N/A")
        out.write(f"[{i}] {law} - المادة {art} (Score: {score})\n")
        out.write(f"    Text: {res.get('text_original', '')[:150]}...\n")

print("Done. Check scratch/retrieval_verify_results.txt")
