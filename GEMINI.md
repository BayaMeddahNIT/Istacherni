Phase 3: Rank-1 Legal Accuracy Protocol
Primary Metric: MRR. We need the correct article at #1, not just in the top 5.

Cross-Code Sensitivity: If a query is criminal, the model must penalize Commercial/Civil results even if the text matches.

Training Data: prioritize the 70 triplets in hard_negatives.jsonl.

Learning Rate: Start at 2e-5; if loss doesn't decrease, adjust to 1e-5.