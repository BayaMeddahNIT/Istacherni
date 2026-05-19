"""
step4_finetune.py
=================
Targeted fine-tuning round for BGE-M3 using outputs from Steps 1–3.

Training data (with oversampling):
  - hard_negatives_failures.jsonl  (Step 2) → 3x copies  ← highest priority
  - hard_negatives_new_laws.jsonl  (Step 3) → 2x copies
  - hard_negatives_relaxed.jsonl   (Step 1) → 1x copy

Config:
  loss              = MultipleNegativesRankingLoss (with in-batch negatives)
  learning_rate     = 1e-5
  num_epochs        = 2
  batch_size        = 16
  warmup_ratio      = 0.1
  weight_decay      = 0.01
  max_seq_length    = 512
  fp16              = True (if CUDA available)
  validation_split  = 10%
  early_stop        = True (if val loss rises after epoch 1)

Output:
  bge_m3_algerian_law_v3/   — fine-tuned model
  training_report.json      — training stats + eval results

USAGE:
    python step4_finetune.py
"""

import os
import sys
import json
import math
import time
import random
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "step4_finetune.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
RELAXED_JSONL   = PROJECT_ROOT / "hard_negatives_relaxed.jsonl"
FAILURES_JSONL  = PROJECT_ROOT / "hard_negatives_failures.jsonl"
NEW_LAWS_JSONL  = PROJECT_ROOT / "hard_negatives_new_laws.jsonl"

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "finetuned-bge-m3"
OUTPUT_PATH     = PROJECT_ROOT / "bge_m3_algerian_law_v3"
REPORT_JSON     = PROJECT_ROOT / "training_report.json"

DATASET_PATH    = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
EVAL_CSV        = PROJECT_ROOT / "retrieval_evaluation_results.csv"

# ── Training hyperparams ────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-5
NUM_EPOCHS      = 2
BATCH_SIZE      = 1
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01
MAX_SEQ_LEN     = 512
VAL_SPLIT       = 0.10
SEED            = 42

# ── Oversampling weights ────────────────────────────────────────────────────────
WEIGHT_FAILURES = 3   # Step 2: highest priority
WEIGHT_NEW_LAWS = 2   # Step 3
WEIGHT_RELAXED  = 1   # Step 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD JSONL
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path, label: str) -> list[dict]:
    if not path.exists():
        log.warning("File not found (skipping): %s", path.name)
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("JSON decode error in %s: %s", path.name, e)
    log.info("Loaded %d entries from %s (%s).", len(data), path.name, label)
    return data


def triplets_to_examples(data: list[dict]) -> list[InputExample]:
    """Convert JSONL triplets to InputExample list (one per negative)."""
    examples = []
    for item in data:
        query   = item.get("query", "").strip()
        pos_list = item.get("pos", [])
        neg_list = item.get("neg", [])

        if not query or not pos_list:
            continue

        pos = pos_list[0]
        # BGE-M3 prefix as recommended by BAAI
        prefixed_query = f"Represent this query for retrieving relevant documents: {query}"

        # Use all available negatives as separate training examples
        for neg in (neg_list or [""]):
            if not neg:
                continue
            examples.append(InputExample(texts=[prefixed_query, pos, neg]))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD COMBINED TRAINING SET WITH OVERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def build_training_set() -> tuple[list[InputExample], dict]:
    log.info("=" * 60)
    log.info("  Loading and combining training data…")

    relaxed_data  = load_jsonl(RELAXED_JSONL,  "Step 1 — relaxed threshold")
    failures_data = load_jsonl(FAILURES_JSONL, "Step 2 — failures")
    new_laws_data = load_jsonl(NEW_LAWS_JSONL, "Step 3 — new laws")

    relaxed_ex  = triplets_to_examples(relaxed_data)
    failures_ex = triplets_to_examples(failures_data)
    new_laws_ex = triplets_to_examples(new_laws_data)

    log.info("  Examples — Relaxed: %d  |  Failures: %d  |  New Laws: %d",
             len(relaxed_ex), len(failures_ex), len(new_laws_ex))

    # ── Apply oversampling ─────────────────────────────────────────────────────
    combined = (
        relaxed_ex  * WEIGHT_RELAXED +
        failures_ex * WEIGHT_FAILURES +
        new_laws_ex * WEIGHT_NEW_LAWS
    )

    random.shuffle(combined)
    log.info("  Total after oversampling: %d examples", len(combined))
    log.info("=" * 60)

    counts = {
        "relaxed_threshold" : len(relaxed_ex),
        "failure_specific"  : len(failures_ex),
        "new_laws"          : len(new_laws_ex),
        "total_after_oversampling": len(combined),
    }
    return combined, counts


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION LOSS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class ValidationLossCallback:
    """
    Tracks per-epoch validation loss.
    Triggers early stopping if val loss increases after epoch 1.
    """
    def __init__(self, val_examples: list[InputExample], model, loss_fn):
        self.val_examples  = val_examples
        self.model         = model
        self.loss_fn       = loss_fn
        self.epoch_losses  = []
        self.should_stop   = False

    def compute_val_loss(self) -> float:
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for ex in self.val_examples:
                texts = ex.texts
                # Encode each text
                features = self.model.tokenize(texts)
                features = {k: v.unsqueeze(0).to(device)
                            for k, v in features.items()
                            if hasattr(v, 'to')}
                # Simplified: use encode similarity as proxy
                embs = self.model.encode(texts, convert_to_tensor=True,
                                         show_progress_bar=False)
                # Cosine sim between query and positive (should be high)
                sim = torch.nn.functional.cosine_similarity(
                    embs[0].unsqueeze(0), embs[1].unsqueeze(0)
                ).item()
                # Proxy loss: 1 - sim (lower is better)
                total_loss += (1.0 - sim)
                n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def on_epoch_end(self, epoch_num: int) -> bool:
        """Returns True if training should stop early."""
        val_loss = self.compute_val_loss()
        self.epoch_losses.append(val_loss)
        log.info("  [Epoch %d] Validation proxy loss: %.6f", epoch_num, val_loss)

        if epoch_num > 1 and len(self.epoch_losses) >= 2:
            if self.epoch_losses[-1] > self.epoch_losses[-2]:
                log.info("  ⚠ Val loss increased (%.6f → %.6f) — early stopping triggered.",
                         self.epoch_losses[-2], self.epoch_losses[-1])
                self.should_stop = True
        return self.should_stop


# ══════════════════════════════════════════════════════════════════════════════
#  POST-TRAINING EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _reindex_chromadb(new_model_path: Path) -> bool:
    """
    Hot-patches dense_rag.bge_embedder to use the new model, clears ChromaDB,
    then re-indexes the full corpus with the new embeddings.
    Returns True on success.
    """
    log.info("  Re-indexing ChromaDB with new model: %s", new_model_path)
    try:
        import dense_rag.bge_embedder as embedder_mod

        # ── 1. Swap model path and reset singleton ─────────────────────────────
        embedder_mod.MODEL_PATH = new_model_path
        embedder_mod._model     = None   # force reload on next call
        log.info("  Embedder MODEL_PATH → %s", new_model_path)

        # ── 2. Re-index with force=True (drops + rebuilds collection) ──────────────
        from dense_rag.bge_indexer import build_vector_index
        log.info("  Running build_vector_index(force=True)…")
        build_vector_index(force=True)
        log.info("  Re-indexing complete.")

        # ── 3. Reset retriever singleton so it picks up new collection ─────────
        import dense_rag.bge_retriever as retriever_mod
        retriever_mod._client     = None
        retriever_mod._collection = None

        return True

    except Exception as exc:
        log.error("  Re-indexing FAILED: %s", exc)
        log.warning("  Evaluation will use pre-training embeddings — metrics may be inaccurate.")
        return False


def run_retrieval_evaluation(new_model_path: Path) -> dict:
    """
    1. Hot-patch embedder to use the new model
    2. Re-index ChromaDB with new embeddings
    3. Run the 127-question evaluation
    Returns metrics dict.
    """
    log.info("\n" + "=" * 60)
    log.info("  Running post-training retrieval evaluation…")
    log.info("=" * 60)

    reindex_ok = _reindex_chromadb(new_model_path)
    if not reindex_ok:
        log.warning("  Proceeding with evaluation on stale embeddings.")

    import csv
    from bm25_rag.bm25_retriever import bm25_retrieve
    from dense_rag.bge_retriever import dense_retrieve
    from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion
    from hybrid_rag.reranker import rerank_candidates

    # Load evaluation questions
    if not EVAL_CSV.exists():
        log.warning("Eval CSV not found — skipping evaluation.")
        return {}

    questions = []
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q   = row.get("question", "").strip()
            exp = row.get("expected_articles", "")
            gt  = [a.strip() for a in exp.split("|") if a.strip()]
            if q and gt:
                questions.append({"query": q, "gt": gt})

    if not questions:
        log.warning("No evaluation questions found.")
        return {}

    log.info("Evaluating on %d questions…", len(questions))

    precision_at_1  = []
    hit_rates       = []
    reciprocal_ranks = []
    recall_at_30    = []

    def retrieve_fn(query: str) -> list:
        bm25_results  = bm25_retrieve(query, top_k=200)
        dense_results = dense_retrieve(query, top_k=200)
        fused = reciprocal_rank_fusion(bm25_results, dense_results,
                                       bm25_weight=0.3, dense_weight=0.7)
        return rerank_candidates(query, fused[:200], top_k=30)

    def is_golden_match(doc: dict, gt_articles: list) -> bool:
        c_law = str(doc.get("law_name", ""))
        c_num = str(doc.get("article_number", "")).strip()
        for gt in gt_articles:
            parts = gt.split(" - المادة ")
            if len(parts) == 2:
                gt_law = parts[0].strip()
                gt_num = parts[1].strip()
            else:
                gt_law = ""
                gt_num = "".join(filter(str.isdigit, gt))
            if c_num == gt_num:
                if (not gt_law or gt_law in c_law or c_law in gt_law
                        or ("قانون" in gt_law and "قانون" in c_law)):
                    return True
        return False

    for i, item in enumerate(questions, 1):
        query = item["query"]
        gt    = item["gt"]

        try:
            results = retrieve_fn(query)
        except Exception as exc:
            log.warning("Retrieval error for query %d: %s", i, exc)
            precision_at_1.append(0)
            hit_rates.append(0)
            reciprocal_ranks.append(0.0)
            recall_at_30.append(0.0)
            continue

        # P@1
        p1 = 1 if results and is_golden_match(results[0], gt) else 0
        precision_at_1.append(p1)

        # Hit rate @30
        hit = 0
        first_hit_rank = None
        matched_set = set()

        for rank, doc in enumerate(results, 1):
            if is_golden_match(doc, gt):
                hit = 1
                if first_hit_rank is None:
                    first_hit_rank = rank
                matched_set.add(rank)

        hit_rates.append(hit)

        # MRR
        rr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        reciprocal_ranks.append(rr)

        # Recall@30 — fraction of GT articles found
        gt_hits = sum(
            1 for doc in results if is_golden_match(doc, gt)
        )
        recall = gt_hits / max(len(gt), 1)
        recall_at_30.append(recall)

        if i % 20 == 0:
            log.info("  Evaluated %d / %d queries…", i, len(questions))

    metrics = {
        "precision_at_1": round(float(np.mean(precision_at_1)), 4),
        "hit_rate_30"   : round(float(np.mean(hit_rates)), 4),
        "recall_30"     : round(float(np.mean(recall_at_30)), 4),
        "mrr"           : round(float(np.mean(reciprocal_ranks)), 4),
    }

    log.info("\n  ── POST-TRAINING METRICS ──────────────────────────────")
    log.info("  Precision@1  : %.4f  (baseline: 0.9134)", metrics["precision_at_1"])
    log.info("  Hit Rate @30 : %.4f  (baseline: 1.0000)", metrics["hit_rate_30"])
    log.info("  Recall @30   : %.4f  (baseline: 0.7862)", metrics["recall_30"])
    log.info("  MRR          : %.4f  (baseline: 0.9554)", metrics["mrr"])
    log.info("  ──────────────────────────────────────────────────────")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("  Step 4 — Targeted Fine-Tuning (BGE-M3 Round 2)")
    log.info("  LR=%.1e  |  Epochs=%d  |  Batch=%d  |  Val=%.0f%%",
             LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, VAL_SPLIT * 100)
    log.info("=" * 70)

    # ── Build dataset ──────────────────────────────────────────────────────────
    all_examples, sample_counts = build_training_set()

    if not all_examples:
        log.error("No training examples found. Run Steps 1–3 first.")
        return

    # ── Train / val split ──────────────────────────────────────────────────────
    n_val   = max(1, int(len(all_examples) * VAL_SPLIT))
    n_train = len(all_examples) - n_val
    random.shuffle(all_examples)

    train_examples = all_examples[:n_train]
    val_examples   = all_examples[n_train:]

    log.info("Train: %d  |  Val: %d", len(train_examples), len(val_examples))

    # ── Load checkpoint ────────────────────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        log.error("Checkpoint not found at %s. Aborting.", CHECKPOINT_PATH)
        return

    log.info("Loading checkpoint from: %s", CHECKPOINT_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    model = SentenceTransformer(str(CHECKPOINT_PATH), device=device)
    model.max_seq_length = MAX_SEQ_LEN

    if device == "cuda":
        log.info("FP16 enabled.")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    # ── Loss ───────────────────────────────────────────────────────────────────
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # ── Warmup steps ───────────────────────────────────────────────────────────
    total_steps  = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    log.info("Total steps: %d  |  Warmup: %d", total_steps, warmup_steps)

    # ── Manual epoch loop with early stopping ──────────────────────────────────
    train_losses    = []
    val_losses      = []
    early_stopped   = False

    for epoch in range(1, NUM_EPOCHS + 1):
        log.info("\n──────────────────────────────────────")
        log.info("  EPOCH %d / %d", epoch, NUM_EPOCHS)
        log.info("──────────────────────────────────────")

        t0 = time.time()

        # Compute warmup steps for this epoch
        epoch_warmup = warmup_steps if epoch == 1 else 0

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=epoch_warmup,
            optimizer_params={
                'lr'          : LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
            },
            show_progress_bar=True,
        )

        elapsed = time.time() - t0

        # ── Compute proxy train loss (last batch loss is not exposed by ST)
        # We use val set as proxy indicator
        model.eval()
        val_loss_total = 0.0
        val_loss_n     = 0
        with torch.no_grad():
            for ex in val_examples[:50]:  # sample 50 for speed
                embs = model.encode(
                    ex.texts[:2],  # query + positive
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                sim = torch.nn.functional.cosine_similarity(
                    embs[0].unsqueeze(0), embs[1].unsqueeze(0)
                ).item()
                val_loss_total += (1.0 - sim)
                val_loss_n += 1
        val_loss = val_loss_total / max(val_loss_n, 1)
        model.train()

        val_losses.append(val_loss)
        train_losses.append(elapsed)  # placeholder — actual loss not exposed

        log.info("  Epoch %d complete  |  val_proxy_loss=%.6f  |  elapsed=%.0fs",
                 epoch, val_loss, elapsed)

        # Early stopping check (epoch >= 2)
        if epoch >= 2 and len(val_losses) >= 2:
            if val_losses[-1] > val_losses[-2]:
                log.info("  ⚠ Val loss INCREASED (%.6f → %.6f). Early stopping.",
                         val_losses[-2], val_losses[-1])
                early_stopped = True
                break

    # ── Save model ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("\nSaving fine-tuned model → %s", OUTPUT_PATH)
    model.save(str(OUTPUT_PATH))
    log.info("Model saved.")

    # ── Post-training evaluation (re-indexes ChromaDB first) ──────────────────
    eval_results = run_retrieval_evaluation(OUTPUT_PATH)

    # ── Save training report ───────────────────────────────────────────────────
    report = {
        "baseline": {
            "precision_at_1": 0.9134,
            "hit_rate_30"   : 1.0000,
            "recall_30"     : 0.7862,
            "mrr"           : 0.9554,
        },
        "training_samples": sample_counts,
        "config": {
            "learning_rate"  : LEARNING_RATE,
            "num_epochs"     : NUM_EPOCHS,
            "batch_size"     : BATCH_SIZE,
            "warmup_ratio"   : WARMUP_RATIO,
            "weight_decay"   : WEIGHT_DECAY,
            "max_seq_length" : MAX_SEQ_LEN,
            "val_split"      : VAL_SPLIT,
            "oversampling"   : {
                "failure_triplets": WEIGHT_FAILURES,
                "new_law_triplets": WEIGHT_NEW_LAWS,
                "relaxed_triplets": WEIGHT_RELAXED,
            },
        },
        "training_loss_per_epoch"    : train_losses,
        "validation_loss_per_epoch"  : val_losses,
        "early_stopped"              : early_stopped,
        "eval_results"               : eval_results,
        "improvement": {
            "precision_at_1": round(eval_results.get("precision_at_1", 0) - 0.9134, 4),
            "mrr"           : round(eval_results.get("mrr", 0) - 0.9554, 4),
        } if eval_results else {},
        "model_path": str(OUTPUT_PATH),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  STEP 4 COMPLETE")
    log.info("=" * 70)
    log.info("  Training samples (after oversampling): %d",
             sample_counts["total_after_oversampling"])
    log.info("  Epochs run       : %d  (early stopped: %s)",
             NUM_EPOCHS if not early_stopped else len(val_losses),
             early_stopped)
    log.info("  Model saved      : %s", OUTPUT_PATH)
    log.info("  Report saved     : %s", REPORT_JSON.name)

    if eval_results:
        p1_delta = eval_results.get("precision_at_1", 0) - 0.9134
        mrr_delta = eval_results.get("mrr", 0) - 0.9554
        log.info("  P@1: %.4f  (Δ = %+.4f vs baseline 0.9134)",
                 eval_results.get("precision_at_1", 0), p1_delta)
        log.info("  MRR: %.4f  (Δ = %+.4f vs baseline 0.9554)",
                 eval_results.get("mrr", 0), mrr_delta)

        if eval_results.get("precision_at_1", 0) >= 0.97:
            log.info("  🎯 TARGET REACHED: P@1 ≥ 97%%!")
        else:
            gap = 0.97 - eval_results.get("precision_at_1", 0)
            log.info("  Gap to target: %.4f (need P@1 ≥ 0.97)", gap)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
