# Final Combined RAG Evaluation Report

## Model Comparison

| Model | Similarity | Correctness | Faithfulness | Article Recall |
|------|------------|-------------|--------------|----------------|
| Agentic RAG + bge | 1.0000 | 1.0000 | 0.9020 | 0.9450 |
| Agentic RAG + qwen embeddings | 1.0000 | 1.0000 | 0.8750 | 0.8850 |
| Agentic RAG + camelbert | 1.0000 | 1.0000 | 0.8490 | 0.8250 |
| Graph RAG + bge | 0.9462 | 0.9162 | 0.1250 | 0.8100 |
| Graph RAG + qwen embeddings | 0.9462 | 0.9162 | 0.1100 | 0.7300 |
| Graph RAG + camelbert | 0.9462 | 0.9162 | 0.0950 | 0.6700 |

## Ranking Tables

### Best Pipeline Ranking
| Rank | Pipeline | Similarity | Correctness | Faithfulness | Article Recall | Avg |
|------|----------|------------|-------------|--------------|----------------|-----|
| 1 | Agentic RAG | 1.0000 | 1.0000 | 0.8753 | 0.8850 | 0.9401 |
| 2 | Graph RAG | 0.9462 | 0.9162 | 0.1100 | 0.7367 | 0.6773 |

### Best Embedding Model Ranking
| Rank | Embedding Model | Similarity | Correctness | Faithfulness | Article Recall | Avg |
|------|-----------------|------------|-------------|--------------|----------------|-----|
| 1 | BGE | 0.9731 | 0.9581 | 0.5135 | 0.8775 | 0.8306 |
| 2 | QWEN EMBEDDINGS | 0.9731 | 0.9581 | 0.4925 | 0.8075 | 0.8078 |
| 3 | CAMELBERT | 0.9731 | 0.9581 | 0.4720 | 0.7475 | 0.7877 |

### Best Overall Configuration Ranking
| Rank | Configuration | Similarity | Correctness | Faithfulness | Article Recall | Avg |
|------|---------------|------------|-------------|--------------|----------------|-----|
| 1 | Agentic RAG + bge | 1.0000 | 1.0000 | 0.9020 | 0.9450 | 0.9617 |
| 2 | Agentic RAG + qwen embeddings | 1.0000 | 1.0000 | 0.8750 | 0.8850 | 0.9400 |
| 3 | Agentic RAG + camelbert | 1.0000 | 1.0000 | 0.8490 | 0.8250 | 0.9185 |
| 4 | Graph RAG + bge | 0.9462 | 0.9162 | 0.1250 | 0.8100 | 0.6994 |
| 5 | Graph RAG + qwen embeddings | 0.9462 | 0.9162 | 0.1100 | 0.7300 | 0.6756 |
| 6 | Graph RAG + camelbert | 0.9462 | 0.9162 | 0.0950 | 0.6700 | 0.6569 |

## Summary
- Best Overall Configuration: Agentic RAG + bge
- Best Pipeline: Agentic RAG
- Best Embedding Model: BGE
