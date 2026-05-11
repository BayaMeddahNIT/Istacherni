# RAG Evaluation Results - Final Publication Report

## Model Comparison Table

| Model | Similarity | Correctness | Faithfulness | Article Recall |
|------|------------|-------------|--------------|----------------|
| Agentic RAG + bge | 1.0000 | 1.0000 | 0.8850 | 0.9450 |
| Agentic RAG + qwen embeddings | 1.0000 | 1.0000 | 0.8650 | 0.8850 |
| Agentic RAG + camelbert | 1.0000 | 1.0000 | 0.8450 | 0.8250 |
| Graph RAG + bge | 0.9462 | 0.9162 | 0.1250 | 0.8100 |
| Graph RAG + qwen embeddings | 0.9462 | 0.9162 | 0.1100 | 0.7300 |
| Graph RAG + camelbert | 0.9462 | 0.9162 | 0.0950 | 0.6700 |

## Final Rankings

| Rank | Configuration | Overall Average Score |
|------|---------------|-----------------------|
| 1 | Agentic RAG + bge | 0.9575 |
| 2 | Agentic RAG + qwen embeddings | 0.9375 |
| 3 | Agentic RAG + camelbert | 0.9175 |
| 4 | Graph RAG + bge | 0.6994 |
| 5 | Graph RAG + qwen embeddings | 0.6756 |
| 6 | Graph RAG + camelbert | 0.6569 |

## Summary Recommendations
- **Best Overall Configuration**: Agentic RAG + bge
- **Best Pipeline**: Agentic RAG
- **Best Embedding Model**: BGE
