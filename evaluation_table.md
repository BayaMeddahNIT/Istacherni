# RAG Evaluation Results

## Model Comparison

| Model | Similarity | Correctness | Faithfulness | Article Recall |
|------|------------|-------------|--------------|----------------|
| Agentic RAG + bge | 0.1594 | 0.1394 | 0.8800 | 0.9200 |
| Agentic RAG + qwen embeddings | 0.1594 | 0.1394 | 0.8600 | 0.8600 |
| Agentic RAG + camelbert | 0.1577 | 0.1411 | 0.8400 | 0.8000 |
| Graph RAG + bge | 0.1594 | 0.1394 | 0.0000 | 0.7800 |
| Graph RAG + qwen embeddings | 0.1594 | 0.1394 | 0.0000 | 0.7000 |
| Graph RAG + camelbert | 0.1577 | 0.1411 | 0.0000 | 0.6400 |

## Summary
- Best Model: Agentic RAG + bge
- Best Pipeline: Agentic RAG
- Best Embedding: bge
