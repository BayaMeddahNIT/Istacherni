# hybrid_rag/reranker.py
"""
Fused Reranker for Hybrid RAG.

When USE_FUSION=true (default):
  Combines BGE-M3 Unified scores with Fine-Tuned Cross-Encoder scores
  using FUSION_ALPHA weight.

When USE_FUSION=false:
  Falls back to BGE-M3 Unified only (alpha=1.0 behavior).
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

PROJECT_ROOT            = Path(__file__).resolve().parent.parent
FINETUNED_RERANKER_PATH = PROJECT_ROOT / "reranker_finetuned"
BGEM3_UNIFIED_PATH      = str(PROJECT_ROOT / "models" / "bge-m3-unified")

QUERY_PROMPT = "Represent this sentence for searching relevant passages: {query}"

USE_FUSION   = os.environ.get("USE_FUSION", "true").lower() == "true"
FUSION_ALPHA = float(os.environ.get("FUSION_ALPHA", "0.7"))

_bgem3_model = None
_bgem3_device = None

_ce_model = None
_ce_tokenizer = None
_ce_device = None

def _get_bgem3_model():
    global _bgem3_model, _bgem3_device
    if _bgem3_model is not None:
        return _bgem3_model, _bgem3_device
    _bgem3_device = "cpu"  # GPU reserved for LLM
    print("[Reranker] BGE-M3 Unified running on CPU")
    print("[Reranker] GPU reserved exclusively for Gemma2:9b")
    print(f"[Reranker] Loading BGE-M3 Unified from {BGEM3_UNIFIED_PATH} on {_bgem3_device}")
    _bgem3_model = BGEM3FlagModel(
        BGEM3_UNIFIED_PATH,
        use_fp16=False,  # fp16 not supported on CPU
        device="cpu",
        # Disable multi-process pool to avoid ZeroDivisionError when GPU is hidden
        use_gpu=False,
    )
    return _bgem3_model, _bgem3_device

def _get_ce_reranker():
    global _ce_model, _ce_tokenizer, _ce_device
    if _ce_model is not None:
        return _ce_model, _ce_tokenizer, _ce_device
    _ce_device = "cpu"  # GPU reserved for LLM
    print("[Reranker] Fine-Tuned CE running on CPU")
    print(f"[Reranker] Loading Fine-Tuned CE from {FINETUNED_RERANKER_PATH} on {_ce_device}")
    _ce_tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_RERANKER_PATH))
    _ce_model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_RERANKER_PATH), num_labels=1)
    _ce_model.to("cpu")
    _ce_model.eval()
    return _ce_model, _ce_tokenizer, _ce_device

def _build_doc_text(doc: dict) -> str:
    law_name = doc.get("law_name", "")
    art_num  = doc.get("article_number", "")
    title    = doc.get("title", "")
    original = doc.get("text_original", "")
    expl     = doc.get("text_explanation", "")
    summary  = doc.get("summary", "")
    kws_raw  = doc.get("keywords", "")
    kws      = " ".join(kws_raw) if isinstance(kws_raw, list) else str(kws_raw)
    return f"[{law_name} - المادة {art_num}] {title}. {original} {expl} {summary} {kws}".strip()

def normalize(scores):
    if not scores: return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def rerank_candidates(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    # 1. Compute BGE-M3 Unified scores
    bge_model, bge_device = _get_bgem3_model()
    pairs = []
    for doc in candidates:
        text = _build_doc_text(doc)
        pairs.append([query, text])

    scores_dict = bge_model.compute_score_single_device(
        pairs,
        batch_size=32,
        max_query_length=512,
        max_passage_length=512,
        weights_for_different_modes=[0.4, 0.2, 0.4],
        device=bge_device,
    )
    raw_scores_A = scores_dict["colbert+sparse+dense"]
    
    do_fusion = USE_FUSION and FINETUNED_RERANKER_PATH.exists()

    if not do_fusion:
        for doc, score in zip(candidates, raw_scores_A):
            doc["rerank_score"] = float(score)
            doc["bgem3_score"] = float(score)
            doc["ce_score"] = 0.0
    else:
        # 2. Compute Fine-Tuned CE scores
        ce_model, ce_tokenizer, ce_device = _get_ce_reranker()
        query_prompted = QUERY_PROMPT.format(query=query)
        doc_texts = [_build_doc_text(doc) for doc in candidates]
        batch_size = 16
        raw_scores_B = []

        ce_model.eval()
        with torch.no_grad():
            for i in range(0, len(doc_texts), batch_size):
                bq = [query_prompted] * len(doc_texts[i:i+batch_size])
                bd = doc_texts[i:i+batch_size]
                enc = ce_tokenizer(bq, bd, max_length=512, truncation=True, padding=True, return_tensors="pt").to("cpu")
                out = ce_model(**enc)
                logits = out.logits.squeeze(-1)
                if logits.dim() == 0: logits = logits.unsqueeze(0)
                batch_scores = torch.sigmoid(logits.float()).cpu().numpy().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                raw_scores_B.extend(batch_scores)

        # 3. Normalize and Fuse
        norm_A = normalize(raw_scores_A)
        norm_B = normalize(raw_scores_B)

        for i, doc in enumerate(candidates):
            fused_score = (FUSION_ALPHA * norm_A[i]) + ((1.0 - FUSION_ALPHA) * norm_B[i])
            doc["rerank_score"] = float(fused_score)
            doc["bgem3_score"] = float(norm_A[i])
            doc["ce_score"] = float(norm_B[i])

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
