import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_ROOT / "hard_negatives.jsonl"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "finetuned-bge-m3"

# We fine-tune BGE-M3 (a multilingual dense embedder)
MODEL_NAME = "BAAI/bge-m3"

def load_triplets() -> list[InputExample]:
    examples = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query = data.get("query", "")
            # Prefix query for BGE-M3 as recommended by BAAI
            prefixed_query = f"Represent this query for retrieving relevant documents: {query}"
            
            positives = data.get("pos", [])
            negatives = data.get("neg", [])
            
            if not positives or not negatives:
                continue
                
            pos = positives[0]
            # Use the single hardest negative (usually rank 1 or closest to golden)
            neg = negatives[0]
            
            examples.append(InputExample(texts=[prefixed_query, pos, neg]))
            
    return examples

def main():
    logging.info(f"Loading base model: {MODEL_NAME}")
    # Force CPU or handle GPU if available
    import torch
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    logging.info(f"Loading training data from {TRAIN_FILE}")
    train_examples = load_triplets()
    logging.info(f"Loaded {len(train_examples)} triplets.")
    
    if not train_examples:
        logging.error("No training examples found. Aborting.")
        return
        
    # Reduced batch size to 1 to prevent OOM on 6GB VRAM
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)
    
    # MultipleNegativesRankingLoss requires pairs or triplets (query, pos, neg)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Training Config
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    logging.info("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 2e-5},
        show_progress_bar=True,
    )
    
    logging.info(f"Saving fine-tuned model to {MODEL_SAVE_PATH}")
    model.save(str(MODEL_SAVE_PATH))
    logging.info("Done!")

if __name__ == "__main__":
    main()
