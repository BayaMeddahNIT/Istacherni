#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_reranker_fused.py  —  Script 2 / 3
======================================
Updates hybrid_rag/reranker.py to use fused scoring combining
BGE-M3 Unified (Lexical/Dense) and Fine-tuned CE (Semantic).
Only runs if fuse_reranker.py found an improvement.
"""

import json
import logging
import sys
import os
import shutil
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT     = Path(__file__).resolve().parent
RERANKER_PY      = PROJECT_ROOT / "hybrid_rag" / "reranker.py"
SWEEP_RESULTS    = PROJECT_ROOT / "alpha_sweep_results.json"
FT_MODEL_PATH    = PROJECT_ROOT / "reranker_finetuned"

NEW_RERANKER_CONTENT = '''# hybrid_rag/reranker.py
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
BGEM3_UNIFIED_PATH      = "D:\\\\pfe_baya_models\\\\bge-m3-unified"

QUERY_PROMPT = "Represent this sentence for searching relevant passages: {query}"

USE_FUSION   = os.environ.get("USE_FUSION", "true").lower() == "true"
{alpha_declaration}

_bgem3_model = None
_bgem3_device = None

_ce_model = None
_ce_tokenizer = None
_ce_device = None

def _get_bgem3_model():
    global _bgem3_model, _bgem3_device
    if _bgem3_model is not None:
        return _bgem3_model, _bgem3_device
    _bgem3_device = os.environ.get("RAG_DEVICE", "cpu")
    if _bgem3_device == "cuda" and not torch.cuda.is_available():
        _bgem3_device = "cpu"
    print(f"[Reranker] Loading BGE-M3 Unified from {BGEM3_UNIFIED_PATH} on {_bgem3_device}")
    _bgem3_model = BGEM3FlagModel(BGEM3_UNIFIED_PATH, use_fp16=(_bgem3_device == "cuda"), device=_bgem3_device)
    return _bgem3_model, _bgem3_device

def _get_ce_reranker():
    global _ce_model, _ce_tokenizer, _ce_device
    if _ce_model is not None:
        return _ce_model, _ce_tokenizer, _ce_device
    _ce_device = os.environ.get("RAG_DEVICE", "cpu")
    if _ce_device == "cuda" and not torch.cuda.is_available():
        _ce_device = "cpu"
    print(f"[Reranker] Loading Fine-Tuned CE from {FINETUNED_RERANKER_PATH} on {_ce_device}")
    _ce_tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_RERANKER_PATH))
    _ce_model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_RERANKER_PATH), num_labels=1)
    _ce_model.to(_ce_device)
    if _ce_device == "cuda":
        _ce_model.half()
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

    scores_dict = bge_model.compute_score(pairs, max_passage_length=512, weights_for_different_modes=[0.4, 0.2, 0.4])
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
                enc = ce_tokenizer(bq, bd, max_length=512, truncation=True, padding=True, return_tensors="pt").to(ce_device)
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
'''

def check_sweep_passed() -> float:
    """Verify that fusion improved over BGE-M3 Unified, and return best alpha."""
    if not SWEEP_RESULTS.exists():
        log.error("alpha_sweep_results.json not found — run fuse_reranker.py first.")
        return -1.0

    with open(SWEEP_RESULTS, "r", encoding="utf-8") as f:
        res = json.load(f)

    baseline_p1 = res.get("baseline_bgem3_precision_at_1", 0)
    best_p1     = res.get("best_precision_at_1", 0)
    best_alpha  = res.get("best_alpha", 1.0)
    best_mrr    = res.get("best_mrr", 0)

    log.info("Sweep Results:")
    log.info("  Baseline BGE-M3 P@1 : %.4f", baseline_p1)
    log.info("  Best Fused P@1      : %.4f (Alpha = %.1f)", best_p1, best_alpha)

    if best_p1 > baseline_p1:
        log.info("  Result: PASS — Fusion improves Precision@1")
        return best_alpha
    else:
        # Check if MRR or NDCG improved
        baseline_mrr = next(x["mrr"] for x in res["sweep"] if x["alpha"] == 1.0)
        log.warning("  Result: FAIL — Fusion did not improve Precision@1.")
        if best_mrr > baseline_mrr:
            log.warning("          However, it did improve MRR from %.4f to %.4f.", baseline_mrr, best_mrr)
        log.warning("  Aborting pipeline update.")
        return -1.0


def main():
    log.info("=" * 60)
    log.info("  update_reranker_fused.py")
    log.info("=" * 60)

    if not FT_MODEL_PATH.exists():
        log.error("Fine-tuned model not found at %s", FT_MODEL_PATH)
        sys.exit(1)

    best_alpha = check_sweep_passed()
    if best_alpha < 0:
        sys.exit(1)

    if not RERANKER_PY.exists():
        log.error("hybrid_rag/reranker.py not found at %s", RERANKER_PY)
        sys.exit(1)

    # Back up original
    backup_path = RERANKER_PY.with_suffix(".py.fused_bak")
    shutil.copy2(RERANKER_PY, backup_path)
    log.info("Backed up original -> %s", backup_path.name)

    # Format the content with the best alpha
    alpha_decl = f'FUSION_ALPHA = float(os.environ.get("FUSION_ALPHA", "{best_alpha}"))'
    content = NEW_RERANKER_CONTENT.replace("{alpha_declaration}", alpha_decl)

    # Write new reranker
    with open(RERANKER_PY, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Updated -> %s with FUSION_ALPHA = %.1f", RERANKER_PY, best_alpha)

    log.info("=" * 60)
    log.info("  DONE — Pipeline updated to use Fused Reranker.")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
