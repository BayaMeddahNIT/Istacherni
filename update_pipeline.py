#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_pipeline.py  —  Script 4 / 4
======================================
Run ONLY after evaluate_reranker.py prints [PASS].

Updates hybrid_rag/reranker.py to use the fine-tuned cross-encoder
when USE_FINETUNED_RERANKER=true (env var, default True).

Original BGEM3FlagModel behavior preserved when env var is false.
The rerank_candidates() signature is unchanged.
"""

import json
import logging
import sys
import os
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
EVAL_RESULTS     = PROJECT_ROOT / "evaluation_comparison.json"
FT_MODEL_PATH    = PROJECT_ROOT / "reranker_finetuned"

QUERY_PROMPT = "Represent this sentence for searching relevant passages: {query}"

NEW_RERANKER_CONTENT = '''# hybrid_rag/reranker.py
"""
Cross-Encoder Reranker for Hybrid RAG.

When USE_FINETUNED_RERANKER=true (default):
  Uses the fine-tuned bge-reranker-v2-m3 cross-encoder.

When USE_FINETUNED_RERANKER=false:
  Falls back to original BGEM3FlagModel (unified multi-head scoring).
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

PROJECT_ROOT            = Path(__file__).resolve().parent.parent
FINETUNED_RERANKER_PATH = PROJECT_ROOT / "reranker_finetuned"
BGEM3_UNIFIED_PATH      = "D:\\\\pfe_baya_models\\\\bge-m3-unified"

# Query prompt — must match fine-tuning (Script 2)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: {query}"

USE_FINETUNED = os.environ.get("USE_FINETUNED_RERANKER", "true").lower() == "true"

_model     = None
_tokenizer = None
_device    = None


def _get_model():
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    _device = os.environ.get("RAG_DEVICE", "cpu")
    if _device == "cuda" and not torch.cuda.is_available():
        _device = "cpu"

    if USE_FINETUNED and FINETUNED_RERANKER_PATH.exists():
        print(f"[Reranker] Loading fine-tuned CE from {FINETUNED_RERANKER_PATH} on {_device}")
        _tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_RERANKER_PATH))
        _model     = AutoModelForSequenceClassification.from_pretrained(
                         str(FINETUNED_RERANKER_PATH), num_labels=1)
        _model.to(_device)
        if _device == "cuda":
            _model.half()
        _model.eval()
    else:
        if USE_FINETUNED:
            print(f"[Reranker] Fine-tuned model not found at {FINETUNED_RERANKER_PATH} "
                  f"— falling back to BGE-M3 Unified")
        print(f"[Reranker] Loading BGE-M3 Unified from {BGEM3_UNIFIED_PATH} on {_device}")
        _model     = BGEM3FlagModel(BGEM3_UNIFIED_PATH,
                                    use_fp16=(_device == "cuda"), device=_device)
        _tokenizer = None  # not used for BGEM3FlagModel

    return _model, _tokenizer, _device


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


def rerank_candidates(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    model, tokenizer, device = _get_model()

    if tokenizer is None:
        # BGE-M3 Unified path
        pairs = []
        for doc in candidates:
            text = _build_doc_text(doc)
            pairs.append([query, text])

        scores_dict = model.compute_score(pairs, max_passage_length=512,
                                          weights_for_different_modes=[0.4, 0.2, 0.4])
        scores = scores_dict["colbert+sparse+dense"]
    else:
        # Fine-tuned cross-encoder path
        query_prompted = QUERY_PROMPT.format(query=query)
        doc_texts      = [_build_doc_text(doc) for doc in candidates]
        batch_size     = 16
        scores         = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(doc_texts), batch_size):
                bq  = [query_prompted] * len(doc_texts[i:i + batch_size])
                bd  = doc_texts[i:i + batch_size]
                enc = tokenizer(bq, bd, max_length=512, truncation=True,
                                padding=True, return_tensors="pt").to(device)
                out    = model(**enc)
                batch_scores = out.logits.squeeze(-1).cpu().float().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

    # Attach scores and sort
    for doc, score in zip(candidates, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
'''


def check_evaluation_passed() -> bool:
    """Verify that the last evaluation run produced a [PASS] verdict."""
    if not EVAL_RESULTS.exists():
        log.error("evaluation_comparison.json not found — run evaluate_reranker.py first.")
        return False

    with open(EVAL_RESULTS, "r", encoding="utf-8") as f:
        results = json.load(f)

    p1_bge = results.get("reranker_A_bgem3_unified", {}).get("precision_at_1", 0)
    p1_ft  = results.get("reranker_C_finetuned_ce",  {}).get("precision_at_1", 0)

    log.info("Evaluation results:")
    log.info("  BGE-M3 Unified P@1 : %.4f  (%.2f%%)", p1_bge, p1_bge * 100)
    log.info("  Fine-tuned CE  P@1 : %.4f  (%.2f%%)", p1_ft,  p1_ft  * 100)

    if p1_ft > p1_bge:
        log.info("  Result: PASS — Fine-tuned CE beats BGE-M3 Unified")
        return True
    else:
        log.warning("  Result: FAIL — Fine-tuned CE does NOT beat BGE-M3 Unified")
        log.warning("  Do NOT update the pipeline. Fix training first.")
        return False


def run_smoke_test():
    """Verify the updated pipeline works end-to-end."""
    log.info("Running smoke test...")

    sample_candidates = [
        {
            "law_name": "قانون العقوبات",
            "article_number": "219",
            "title": "النصب والاحتيال",
            "text_original": "يعاقب بالحبس من سنة إلى خمس سنوات",
            "text_explanation": "جريمة النصب",
            "summary": "عقوبة النصب",
            "keywords": ["نصب", "احتيال"],
        },
        {
            "law_name": "القانون المدني",
            "article_number": "86",
            "title": "صحة العقد",
            "text_original": "يجب أن يكون محل الالتزام ممكناً",
            "text_explanation": "شروط صحة العقد",
            "summary": "محل الالتزام",
            "keywords": ["عقد", "التزام"],
        },
    ]

    sys.path.insert(0, str(PROJECT_ROOT))
    from hybrid_rag.reranker import rerank_candidates, USE_FINETUNED

    query  = "هل التزوير في الفواتير يعاقب عليه؟"
    result = rerank_candidates(query, sample_candidates, top_k=5)

    assert len(result) > 0, "rerank_candidates returned empty list"
    assert "rerank_score" in result[0], "rerank_score missing from result"
    log.info("  Smoke test: returned %d candidates", len(result))
    log.info("  Top result: %s Art.%s  score=%.4f",
             result[0].get("law_name"), result[0].get("article_number"),
             result[0].get("rerank_score", 0))
    log.info("  Model in use: %s", "fine-tuned CE" if USE_FINETUNED else "BGE-M3 Unified")
    print("  Pipeline updated successfully — using fine-tuned CE")


def main():
    log.info("=" * 60)
    log.info("  update_pipeline.py  —  Script 4 / 4")
    log.info("=" * 60)

    if not FT_MODEL_PATH.exists():
        log.error("Fine-tuned model not found at %s — run finetune_reranker.py first",
                  FT_MODEL_PATH)
        sys.exit(1)

    if not check_evaluation_passed():
        log.error("Cannot update pipeline — evaluation FAILED.")
        sys.exit(1)

    if not RERANKER_PY.exists():
        log.error("hybrid_rag/reranker.py not found at %s", RERANKER_PY)
        sys.exit(1)

    # Back up original
    backup_path = RERANKER_PY.with_suffix(".py.bak")
    import shutil
    shutil.copy2(RERANKER_PY, backup_path)
    log.info("Backed up original -> %s", backup_path.name)

    # Write new reranker
    with open(RERANKER_PY, "w", encoding="utf-8") as f:
        f.write(NEW_RERANKER_CONTENT)
    log.info("Updated -> %s", RERANKER_PY)

    # Smoke test
    run_smoke_test()

    log.info("=" * 60)
    log.info("  DONE — Pipeline updated to use fine-tuned CE reranker.")
    log.info("  To revert: copy hybrid_rag/reranker.py.bak -> reranker.py")
    log.info("  To disable: set USE_FINETUNED_RERANKER=false in environment")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
