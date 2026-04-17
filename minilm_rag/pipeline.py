"""
pipeline.py
-----------
Convenience single-call entry point for the MiniLM hybrid RAG pipeline.

Usage:
  from minilm_rag.pipeline import minilm_answer

  answer = minilm_answer("ما هي عقوبة السرقة في القانون الجزائري؟")
  print(answer)

Or run directly:
  python -m minilm_rag.pipeline
"""

from __future__ import annotations

from minilm_rag.minilm_retriever import minilm_retrieve
from minilm_rag.minilm_generator import minilm_generate


def minilm_answer(
    question:     str,
    top_k:        int   = 5,
    bm25_weight:  float = 0.4,
    embed_weight: float = 0.6,
    max_retries:  int   = 4,
    verbose:      bool  = False,
) -> str:
    """
    Full hybrid RAG pipeline: retrieve → generate.

    Args:
        question:     User's Arabic/French legal question.
        top_k:        Number of articles to retrieve.
        bm25_weight:  Weight for BM25 in score fusion (default 0.4).
        embed_weight: Weight for cosine-sim in score fusion (default 0.6).
        max_retries:  Gemini API retry count on rate-limit.
        verbose:      If True, print retrieved article titles before answering.

    Returns:
        Arabic answer string grounded in retrieved articles.
    """
    chunks = minilm_retrieve(
        question,
        top_k=top_k,
        bm25_weight=bm25_weight,
        embed_weight=embed_weight,
    )

    if verbose:
        print(f"\n[MiniLM Pipeline] Retrieved {len(chunks)} articles:")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. {c['law_name']} — المادة {c['article_number']}  "
                  f"(hybrid={c['hybrid_score']}  "
                  f"bm25={c['bm25_score']}  "
                  f"embed={c['embed_score']})")
        print()

    return minilm_generate(question, chunks, max_retries=max_retries)


# ── CLI demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_questions = [
        #"ما هي عقوبة السرقة في القانون الجزائري؟",
        #"ما هي عقوبة التهديد بالعنف؟",
        #"ما هو تعريف القذف والتشهير؟",
        "هل يمكن للشريك في شركة SNC أن يتنازل عن حصصه لزوجته دون موافقة البقية؟"
    ]

    for q in demo_questions:
        print(f"\n{'='*65}")
        print(f"❓  {q}")
        print("=" * 65)
        ans = minilm_answer(q, top_k=5, verbose=True)
        print(ans)
