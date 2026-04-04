# RAG Architecture Evaluation Report

## Summary

| Model | Recall@1 | Recall@3 | Recall@5 | MRR | Faithfulness | Relevance | Completeness | **Accuracy** |
|---|---|---|---|---|---|---|---|---|
| BM25 RAG (Lexical) | 40.0% | 60.0% | 80.0% | 0.507 | 0.00 | 0.00 | 0.00 | **32.0%** |
| Graph RAG | 20.0% | 40.0% | 60.0% | 0.350 | 0.00 | 0.00 | 0.00 | **24.0%** |

## 🏆 Winner

**BM25 RAG (Lexical)** with an accuracy of **32.0%**


## Accuracy Formula

```
accuracy = Recall@5 × 0.40  +  GenScore × 0.60

```
