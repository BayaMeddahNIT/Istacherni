# hybrid_rag/reranker.py
"""
Cross-Encoder Reranker for Hybrid RAG.
Takes candidates from BM25 and Dense retrievers and re-scores them for higher accuracy.
"""

import os
import torch
from pathlib import Path
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_RERANKER_PATH = PROJECT_ROOT / "models" / "finetuned-cross-encoder"

# Use Unified Multi-head BGE-M3 model as reranker
RERANKER_MODEL_NAME = "D:\\pfe_baya_models\\bge-m3-unified"

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        # Use GPU for fast indexing and evaluation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Unified BGE-M3 Reranker on {device}: {RERANKER_MODEL_NAME} ...")
        # Load the fine-tuned Multi-head model
        _reranker = BGEM3FlagModel(
            RERANKER_MODEL_NAME, 
            use_fp16=True if device == "cuda" else False,
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

    # Predict the relevance scores using unified multi-head scoring
    # Weights: Dense(0.4), Sparse(0.2), ColBERT(0.4)
    scores_dict = reranker.compute_score(pairs, max_passage_length=512, weights_for_different_modes=[0.4, 0.2, 0.4])
    scores = scores_dict["colbert+sparse+dense"]

    # Attach the new scores to the candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # Sort candidates by the new rerank_score in descending order
    reranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # Return the top K candidates
    return reranked_candidates[:top_k]
