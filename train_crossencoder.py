import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_ROOT / "hard_negatives.jsonl"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "finetuned-cross-encoder"

# BAAI/bge-reranker-v2-m3 is a cross-encoder
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

class TripletCrossEncoderDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                query = data.get("query", "")
                
                positives = data.get("pos", [])
                negatives = data.get("neg", [])
                
                if not positives or not negatives:
                    continue
                    
                pos = positives[0]
                neg = negatives[0]
                
                self.examples.append((query, pos, neg))
                
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        query, pos, neg = self.examples[idx]
        
        # Tokenize (query, pos)
        pos_enc = self.tokenizer(
            query, pos, 
            truncation=True, max_length=self.max_length, 
            padding="max_length", return_tensors="pt"
        )
        
        # Tokenize (query, neg)
        neg_enc = self.tokenizer(
            query, neg, 
            truncation=True, max_length=self.max_length, 
            padding="max_length", return_tensors="pt"
        )
        
        return {
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading CrossEncoder: {MODEL_NAME} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(device)
    
    logging.info(f"Loading dataset from {TRAIN_FILE}")
    dataset = TripletCrossEncoderDataset(TRAIN_FILE, tokenizer)
    # Reduced batch size to 1 to prevent OOM
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    logging.info(f"Loaded {len(dataset)} triplets.")
    
    if len(dataset) == 0:
        logging.error("No training data found.")
        return
        
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    # Margin Ranking Loss with margin = 0.3
    # loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    # y = 1 implies we want x1 (pos) to be greater than x2 (neg) by at least margin
    margin = 0.3
    loss_fn = torch.nn.MarginRankingLoss(margin=margin)
    target = torch.ones(dataloader.batch_size).to(device)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass for positive pairs
            pos_outputs = model(
                input_ids=batch["pos_input_ids"].to(device),
                attention_mask=batch["pos_attention_mask"].to(device)
            )
            pos_scores = pos_outputs.logits.squeeze(-1)
            
            # Forward pass for negative pairs
            neg_outputs = model(
                input_ids=batch["neg_input_ids"].to(device),
                attention_mask=batch["neg_attention_mask"].to(device)
            )
            neg_scores = neg_outputs.logits.squeeze(-1)
            
            # Adjust target size if last batch is smaller
            curr_target = target[:pos_scores.size(0)]
            
            # Compute loss: we want pos_scores > neg_scores by at least 0.3
            loss = loss_fn(pos_scores, neg_scores, curr_target)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        logging.info(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(dataloader):.4f}")
        
    logging.info(f"Saving fine-tuned CrossEncoder to {MODEL_SAVE_PATH}")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    logging.info("Done!")

if __name__ == "__main__":
    main()
