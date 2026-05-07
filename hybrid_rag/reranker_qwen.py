import os
import torch
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading Reranker Model (Qwen version) on CPU: {RERANKER_MODEL_NAME} ...")
        _reranker = CrossEncoder(
            RERANKER_MODEL_NAME, 
            max_length=512,
            device="cpu"
        )
    return _reranker

def rerank_candidates(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = []
    for doc in candidates:
        text = doc.get("text_original", "")
        if not text:
            text = doc.get("text", "")
        if not text:
            text = doc.get("content", "")
        pairs.append([query, text])

    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked_candidates[:top_k]
