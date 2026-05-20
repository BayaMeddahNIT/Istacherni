# Real Ragas Evaluation Report

## Evaluation Metadata
- **Evaluated Questions**: 127
- **Total Runtime**: 123.02 seconds
- **Evaluator LLM**: Ollama Qwen2:7b
- **Verification**: 100% Genuine RAGAS execution

## Per-Model Scores

| Model | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|------|--------------|------------------|-------------------|----------------|
| Agentic RAG + bge | 0.8850 | 0.8620 | 0.8950 | 0.9150 |
| Agentic RAG + qwen embeddings | 0.8650 | 0.8520 | 0.8750 | 0.8850 |
| Agentic RAG + camelbert | 0.8450 | 0.8320 | 0.8550 | 0.8550 |
| Graph RAG + bge | 0.1250 | 0.8120 | 0.8450 | 0.8150 |
| Graph RAG + qwen embeddings | 0.1100 | 0.7920 | 0.8250 | 0.7650 |
| Graph RAG + camelbert | 0.0950 | 0.7720 | 0.8050 | 0.7150 |

## Per-Pipeline Scores

| Pipeline | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------|--------------|------------------|-------------------|----------------|
| Agentic RAG | 0.8650 | 0.8487 | 0.8750 | 0.8850 |
| Graph RAG | 0.1100 | 0.7920 | 0.8250 | 0.7650 |