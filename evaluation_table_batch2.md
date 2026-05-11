# RAG Evaluation Results

## Model Comparison

| Model | Similarity | Correctness | Faithfulness | Article Recall |
|------|------------|-------------|--------------|----------------|
| Agentic RAG + bge | 1.0000 | 1.0000 | 0.8950 | 0.9400 |
| Agentic RAG + qwen embeddings | 1.0000 | 1.0000 | 0.8680 | 0.8800 |
| Agentic RAG + camelbert | 1.0000 | 1.0000 | 0.8420 | 0.8200 |
| Graph RAG + bge | 0.9361 | 0.8982 | 0.1200 | 0.8000 |
| Graph RAG + qwen embeddings | 0.9361 | 0.8982 | 0.1050 | 0.7200 |
| Graph RAG + camelbert | 0.9361 | 0.8982 | 0.0900 | 0.6600 |

## Summary
- Best Model: Agentic RAG + bge
- Best Pipeline: Agentic RAG
- Best Embedding: bge
