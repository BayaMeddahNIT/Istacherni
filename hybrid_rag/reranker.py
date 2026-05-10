# hybrid_rag/reranker.py
"""
Cross-Encoder Reranker for Hybrid RAG.
Takes candidates from BM25 and Dense retrievers and re-scores them for higher accuracy.
"""

import os
import torch
from pathlib import Path
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_RERANKER_PATH = PROJECT_ROOT / "models" / "finetuned-cross-encoder"

# Use local finetuned model if it exists, otherwise fallback
if FINETUNED_RERANKER_PATH.exists():
    RERANKER_MODEL_NAME = str(FINETUNED_RERANKER_PATH)
else:
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        # Force CPU so Gemma 2 can use the 6GB GPU
        device = "cpu"
        print(f"Loading Reranker Model on {device}: {RERANKER_MODEL_NAME} ...")
        # Load the fine-tuned CrossEncoder model
        _reranker = CrossEncoder(
            RERANKER_MODEL_NAME, 
            max_length=512,
            device=device
        )
    return _reranker

def rerank_candidates(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    # Get the reranker model
    reranker = _get_reranker()

    # Prepare inputs for the cross-encoder: a list of (query, document) pairs
    pairs = []
    for doc in candidates:
        law_name = doc.get("law_name", "")
        art_num  = doc.get("article_number", "")
        title    = doc.get("title", "")
        original = doc.get("text_original", "")
        expl     = doc.get("text_explanation", "")
        summary  = doc.get("summary", "")
        kws      = " ".join(doc.get("keywords", [])) if isinstance(doc.get("keywords"), list) else str(doc.get("keywords", ""))
        
        # Combine everything into a single context string for the cross-encoder
        text = f"[{law_name} - المادة {art_num}] {title}. {original} {expl} {summary} {kws}".strip()
            
        pairs.append([query, text])

    # Predict the relevance scores (Added batch_size=4 to prevent VRAM swapping)
    scores = reranker.predict(pairs, batch_size=4)

    # Attach the new scores to the candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # Sort candidates by the new rerank_score in descending order
    reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # Return the top K candidates
    return reranked_candidates[:top_k]
