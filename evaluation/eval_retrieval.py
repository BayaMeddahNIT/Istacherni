"""
eval_retrieval.py
-----------------
Retrieval evaluation: computes Recall@1, Recall@3, Recall@5, and MRR
for a given retriever function across all test cases.

Retriever function contract:
  fn(query: str, top_k: int) -> list[dict]
  Each returned dict MUST have an "id" key matching article IDs in the corpus.
"""

from dataclasses import dataclass
from evaluation.eval_testset import TestCase


@dataclass
class RetrievalResult:
    tc_id:           str
    question:        str
    domain:          str
    expected_ids:    list[str]
    retrieved_ids:   list[str]
    recall_at_1:     float   # 1.0 or 0.0
    recall_at_3:     float   # 1.0 or 0.0
    recall_at_5:     float   # 1.0 or 0.0
    reciprocal_rank: float   # 1/rank of first correct hit, 0 if none


def _hits_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    return 1.0 if any(eid in top_k for eid in expected_ids) else 0.0


def _reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    model_name: str,
    retriever_fn,       # callable(query: str, top_k: int) -> list[dict]
    test_cases: list[TestCase],
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Evaluate retrieval quality for one model over all test cases.

    Args:
        model_name:   Display name for the model.
        retriever_fn: A function that takes (query, top_k) and returns
                      a list of dicts each with an 'id' key.
        test_cases:   List of TestCase objects from eval_testset.
        top_k:        Maximum number of results to request.
        verbose:      Whether to print per-question results.

    Returns:
        {
          "model":       str,
          "recall_at_1": float,   ← averaged over all test cases
          "recall_at_3": float,
          "recall_at_5": float,
          "mrr":         float,
          "per_case":    list[RetrievalResult]
        }
    """
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Retrieval eval — {model_name}  ({len(test_cases)} queries)")
        print(f"{'─'*60}")

    per_case: list[RetrievalResult] = []

    for tc in test_cases:
        try:
            results = retriever_fn(tc.question, top_k)
            retrieved_ids = [r.get("id", "") for r in results]
        except Exception as e:
            if verbose:
                print(f"  ❌ [{tc.id}] retriever error: {e}")
            retrieved_ids = []

        r1  = _hits_at_k(retrieved_ids, tc.expected_ids, 1)
        r3  = _hits_at_k(retrieved_ids, tc.expected_ids, 3)
        r5  = _hits_at_k(retrieved_ids, tc.expected_ids, 5)
        mrr = _reciprocal_rank(retrieved_ids, tc.expected_ids)

        result = RetrievalResult(
            tc_id          = tc.id,
            question       = tc.question,
            domain         = tc.domain,
            expected_ids   = tc.expected_ids,
            retrieved_ids  = retrieved_ids,
            recall_at_1    = r1,
            recall_at_3    = r3,
            recall_at_5    = r5,
            reciprocal_rank = mrr,
        )
        per_case.append(result)

        if verbose:
            hit = "✅" if r5 else "❌"
            print(f"  {hit} [{tc.id}] R@1={r1:.0f} R@3={r3:.0f} R@5={r5:.0f} MRR={mrr:.3f}"
                  f"  | {tc.question[:45]}…")

    # Aggregate
    n = len(per_case) or 1
    agg = {
        "model":       model_name,
        "recall_at_1": round(sum(r.recall_at_1      for r in per_case) / n, 4),
        "recall_at_3": round(sum(r.recall_at_3      for r in per_case) / n, 4),
        "recall_at_5": round(sum(r.recall_at_5      for r in per_case) / n, 4),
        "mrr":         round(sum(r.reciprocal_rank  for r in per_case) / n, 4),
        "per_case":    [vars(r) for r in per_case],
    }

    if verbose:
        print(f"\n  → Recall@1={agg['recall_at_1']:.1%}  "
              f"Recall@3={agg['recall_at_3']:.1%}  "
              f"Recall@5={agg['recall_at_5']:.1%}  "
              f"MRR={agg['mrr']:.3f}")

    return agg
