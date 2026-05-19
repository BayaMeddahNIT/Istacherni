#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_reranker.py  —  Script 2 / 4
========================================
Fine-tunes BAAI/bge-reranker-v2-m3 on Algerian legal data.

Key features:
  - Query prompt applied CONSISTENTLY in training AND evaluation
  - Weighted BCEWithLogitsLoss for class imbalance
  - CrossEncoderTripletEvaluator (mandatory — legal discrimination signal)
  - OOM auto-recovery: batch=4, grad_accum=8
  - Checkpoints every 50 steps (resumable)
  - Outputs: reranker_finetuned/, training_curves.json, triplet_eval_log.json
"""

import gc
import json
import logging
import sys
import os
import shutil

# Must be set before any CUDA allocation to prevent memory fragmentation OOM.
# Allows CUDA to grow allocations dynamically instead of pre-reserving large blocks.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_reranker.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PAIRS     = PROJECT_ROOT / "reranker_train_pairs.json"
VAL_PAIRS       = PROJECT_ROOT / "reranker_val_pairs.json"
TRIPLET_EVAL    = PROJECT_ROOT / "reranker_triplet_eval.json"
OUTPUT_DIR      = PROJECT_ROOT / "reranker_finetuned"
CURVES_JSON     = PROJECT_ROOT / "training_curves.json"
TRIPLET_LOG     = PROJECT_ROOT / "triplet_eval_log.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME          = "BAAI/bge-reranker-v2-m3"
LR                  = 2e-5
NUM_EPOCHS          = 3
# RTX 3060 Laptop (6 GB VRAM): Adafactor + gradient_checkpointing allows batch=8
# but we use batch=4, grad_accum=8 conservatively for stability.
BATCH_SIZE          = 4
GRAD_ACCUM          = 8
WARMUP_RATIO        = 0.1
WEIGHT_DECAY        = 0.01
MAX_LENGTH          = 512
EVAL_STEPS          = 50
SAVE_STEPS          = 50
LOG_STEPS           = 10
EARLY_STOP_PATIENCE = 5
SEED                = 42

# CRITICAL: Applied consistently in BOTH training and evaluation
QUERY_PROMPT = "Represent this sentence for searching relevant passages: {query}"


# ── GPU memory helper ─────────────────────────────────────────────────────────
def log_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / 1024 ** 3
        total  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        free   = total - torch.cuda.memory_reserved() / 1024 ** 3
        log.info("GPU Memory %s: %.2f GB allocated | %.2f GB free / %.2f GB total",
                 label, alloc, free, total)
    return True


# ── Dataset ───────────────────────────────────────────────────────────────────
class RerankerDataset(Dataset):
    def __init__(self, pairs: list, tokenizer, max_length: int = MAX_LENGTH):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item     = self.pairs[idx]
        query    = QUERY_PROMPT.format(query=item["query"])  # Apply prompt
        document = item["document"]
        label    = float(item["label"])

        enc = self.tokenizer(
            query,
            document,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        enc["labels"] = label
        return enc


# ── CrossEncoderTripletEvaluator ─────────────────────────────────────────────
class CrossEncoderTripletEvaluator:
    """
    Evaluates whether the cross-encoder scores the positive document
    higher than the negative for each anchor query.

    Accuracy = % of triplets where score(anchor, positive) > score(anchor, negative)

    This directly measures legal discrimination ability, independent of BCE loss.
    """

    def __init__(self, triplets: list, tokenizer, device: str,
                 name: str = "legal_triplet"):
        self.anchors   = [t["anchor"]   for t in triplets]
        self.positives = [t["positive"] for t in triplets]
        self.negatives = [t["negative"] for t in triplets]
        self.tokenizer = tokenizer
        self.device    = device
        self.name      = name

    def _score_pairs(self, model, queries: list, documents: list,
                     batch_size: int = 16) -> list:
        """Score (query, document) pairs with the cross-encoder."""
        model.eval()
        all_scores = []
        prompted_q = [QUERY_PROMPT.format(query=q) for q in queries]

        with torch.no_grad():
            for i in range(0, len(prompted_q), batch_size):
                bq = prompted_q[i:i + batch_size]
                bd = documents[i:i + batch_size]
                enc = self.tokenizer(
                    bq, bd,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = model(**enc)
                scores  = outputs.logits.squeeze(-1).cpu().float().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                all_scores.extend(scores)

        return all_scores

    def evaluate(self, model, epoch: int = -1, steps: int = -1,
                 output_path: Optional[str] = None) -> float:
        pos_scores = self._score_pairs(model, self.anchors, self.positives)
        neg_scores = self._score_pairs(model, self.anchors, self.negatives)

        correct  = sum(1 for p, n in zip(pos_scores, neg_scores) if p > n)
        total    = len(self.anchors)
        accuracy = correct / max(total, 1)

        log.info("")
        log.info("[TripletEvaluator] Epoch %d  Step %d", epoch, steps)
        log.info("  Accuracy:                 %.4f  (%d / %d)", accuracy, correct, total)
        log.info("  Correctly ranked pairs:   %d", correct)
        log.info("  Incorrectly ranked pairs: %d", total - correct)

        if epoch >= 1 and accuracy < 0.70:
            log.warning("  Model accuracy %.4f < 0.70 after epoch %d "
                        "— model may not be learning. Check data quality.",
                        accuracy, epoch)

        result = {
            "epoch":  epoch,
            "steps":  steps,
            f"{self.name}_accuracy": round(accuracy, 6),
            f"{self.name}_correct":  correct,
            f"{self.name}_total":    total,
        }

        log_path = Path(output_path) / "triplet_eval_log.json" if output_path else TRIPLET_LOG
        history  = []
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        history.append(result)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return accuracy


# ── Callbacks ─────────────────────────────────────────────────────────────────
class TripletEvalCallback(TrainerCallback):
    """Calls TripletEvaluator before training starts and after each epoch."""

    def __init__(self, evaluator: CrossEncoderTripletEvaluator, output_path: str):
        self.evaluator        = evaluator
        self.output_path      = output_path
        self.epoch_accuracies: list = []

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl,
                       model=None, **kwargs):
        log.info("  Running TripletEvaluator BEFORE training (baseline)...")
        acc = self.evaluator.evaluate(model, epoch=0, steps=0,
                                       output_path=self.output_path)
        self.epoch_accuracies.append(acc)
        log.info("  Baseline triplet accuracy: %.4f", acc)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl,
                     model=None, **kwargs):
        epoch = int(state.epoch)
        acc   = self.evaluator.evaluate(model, epoch=epoch,
                                         steps=state.global_step,
                                         output_path=self.output_path)
        self.epoch_accuracies.append(acc)


class CurvesCallback(TrainerCallback):
    def __init__(self):
        self.train_losses:   list = []
        self.val_losses:     list = []
        self.steps:          list = []
        self.learning_rates: list = []

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append(round(logs["loss"], 6))
            self.steps.append(step)
            self.learning_rates.append(round(logs.get("learning_rate", 0.0), 10))
        if "eval_loss" in logs:
            self.val_losses.append({"step": step, "loss": round(logs["eval_loss"], 6)})

    def save(self, triplet_accuracies: list):
        data = {
            "steps":                       self.steps,
            "train_loss":                  self.train_losses,
            "val_loss":                    self.val_losses,
            "learning_rates":              self.learning_rates,
            "triplet_accuracy_per_epoch":  triplet_accuracies,
        }
        with open(CURVES_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("Training curves saved -> %s", CURVES_JSON.name)


# ── Custom Trainer with weighted BCE loss ─────────────────────────────────────
class RerankerTrainer(Trainer):
    def __init__(self, *args, pos_weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits  = outputs.logits.squeeze(-1)

        pw      = self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        loss    = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Training function ──────────────────────────────────────────────────────────
def run_training(batch_size: int, grad_accum: int):
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  |  batch=%d  grad_accum=%d", device, batch_size, grad_accum)

    # Load data
    log.info("Loading training pairs from %s", TRAIN_PAIRS.name)
    with open(TRAIN_PAIRS, "r", encoding="utf-8") as f:
        train_pairs = json.load(f)

    log.info("Loading validation pairs from %s", VAL_PAIRS.name)
    with open(VAL_PAIRS, "r", encoding="utf-8") as f:
        val_pairs = json.load(f)

    log.info("Train pairs: %d  |  Val pairs: %d", len(train_pairs), len(val_pairs))

    # Load tokenizer and model
    log.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    # Wrap entire GPU work in try/except so model is always freed on failure
    # This ensures the retry in main() always starts with clean CUDA memory.
    try:
        model.to(device)
        log_gpu_memory("after model load")

        # Datasets
        train_ds = RerankerDataset(train_pairs, tokenizer)
        val_ds   = RerankerDataset(val_pairs,   tokenizer)
        collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        # Weighted loss
        pos_count  = sum(1 for p in train_pairs if p["label"] == 1.0)
        neg_count  = len(train_pairs) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32)
        log.info("pos_weight = %.4f  (pos=%d, neg=%d)", pos_weight.item(), pos_count, neg_count)
        if pos_weight.item() > 10.0:
            log.warning("pos_weight %.4f > 10 — data may be severely imbalanced", pos_weight.item())

        # TripletEvaluator (mandatory)
        log.info("Loading TripletEvaluator triplets from %s", TRIPLET_EVAL.name)
        with open(TRIPLET_EVAL, "r", encoding="utf-8") as f:
            triplet_data = json.load(f)
        log.info("TripletEvaluator: %d triplets loaded", len(triplet_data))

        triplet_evaluator = CrossEncoderTripletEvaluator(
            triplet_data, tokenizer, device, name="legal_triplet"
        )

        # Check for existing checkpoint (resume support)
        resume_ckpt = None
        if OUTPUT_DIR.exists():
            ckpts = sorted(OUTPUT_DIR.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]))
            if ckpts:
                resume_ckpt = str(ckpts[-1])
                log.info("Resuming from checkpoint: %s", resume_ckpt)

        # Enable gradient checkpointing to trade compute for activation memory.
        # Critical for fitting bge-reranker-v2-m3 in 6 GB VRAM.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            log.info("Gradient checkpointing enabled.")

        # Training arguments
        # optimizer="adafactor": replaces AdamW (~4.5 GB FP32 moments) with
        # Adafactor factored approximation (~0.5 GB). Mandatory for 6 GB VRAM.
        args = TrainingArguments(
            output_dir                  = str(OUTPUT_DIR),
            learning_rate               = LR,
            num_train_epochs            = NUM_EPOCHS,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = grad_accum,
            warmup_ratio                = WARMUP_RATIO,
            weight_decay                = WEIGHT_DECAY,
            fp16                        = (device == "cuda"),
            optim                       = "adafactor",
            eval_strategy               = "steps",
            eval_steps                  = EVAL_STEPS,
            save_strategy               = "steps",
            save_steps                  = SAVE_STEPS,
            save_total_limit            = 5,
            save_only_model             = True,
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            greater_is_better           = False,
            logging_steps               = LOG_STEPS,
            report_to                   = "none",
            seed                        = SEED,
            dataloader_pin_memory       = False,
        )

        triplet_cb = TripletEvalCallback(triplet_evaluator, str(OUTPUT_DIR))
        curves_cb  = CurvesCallback()

        trainer = RerankerTrainer(
            model         = model,
            args          = args,
            train_dataset = train_ds,
            eval_dataset  = val_ds,
            data_collator = collator,
            callbacks     = [
                triplet_cb,
                curves_cb,
                EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE),
            ],
            pos_weight    = pos_weight,
        )

        trainer.train(resume_from_checkpoint=resume_ckpt)

        log.info("Training complete. Saving best model -> %s", OUTPUT_DIR)
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # Run TripletEvaluator on final best model
        log.info("Running TripletEvaluator on final best model...")
        final_acc = triplet_evaluator.evaluate(
            model, epoch=int(trainer.state.epoch),
            steps=trainer.state.global_step,
            output_path=str(OUTPUT_DIR)
        )
        triplet_cb.epoch_accuracies.append(final_acc)

        log_gpu_memory("after training")
        curves_cb.save(triplet_cb.epoch_accuracies)

        return trainer.state.best_metric

    except Exception as exc:
        # CRITICAL: always free the model from GPU before re-raising.
        # Without this, the caller's OOM retry would find CUDA still full.
        log.warning("run_training encountered an error: %s — freeing GPU memory", type(exc).__name__)
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)   # give CUDA driver time to reclaim memory
        log_gpu_memory("after cleanup")
        raise


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("  finetune_reranker.py  —  Script 2 / 4")
    log.info("  Model : %s", MODEL_NAME)
    log.info("  LR=%.1e  Epochs=%d  Batch=%d  GradAccum=%d  MaxLen=%d",
             LR, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, MAX_LENGTH)
    log.info("=" * 65)

    # Validate input files
    for path in [TRAIN_PAIRS, VAL_PAIRS, TRIPLET_EVAL]:
        if not path.exists():
            log.error("Missing required file: %s — run prepare_reranker_data.py first",
                      path.name)
            sys.exit(1)

    log_gpu_memory("pre-training check")

    # Use batch=4, grad_accum=8 always for 6 GB VRAM.
    # Adafactor + gradient_checkpointing makes this sufficient.
    try:
        best_loss = run_training(BATCH_SIZE, GRAD_ACCUM)

    except torch.cuda.OutOfMemoryError:
        log.warning("CUDA OOM with batch=%d — freeing memory and retrying with batch=2", BATCH_SIZE)

        if OUTPUT_DIR.exists():
            for ck in OUTPUT_DIR.glob("checkpoint-*"):
                shutil.rmtree(ck, ignore_errors=True)
                log.info("  Removed partial checkpoint: %s", ck.name)

        # Last resort: batch=2, grad_accum=16 (same effective batch=32)
        best_loss = run_training(batch_size=2, grad_accum=16)

    log.info("=" * 65)
    log.info("  FINE-TUNING COMPLETE")
    log.info("  Best eval_loss : %.6f", best_loss if best_loss else float("nan"))
    log.info("  Model saved    : %s", OUTPUT_DIR)
    log.info("  Curves saved   : %s", CURVES_JSON.name)
    log.info("  Triplet log    : %s", TRIPLET_LOG.name)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
