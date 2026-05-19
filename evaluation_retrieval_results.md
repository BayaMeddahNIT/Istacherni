# RAG Retrieval Evaluation Results

**Dataset:** 127 questions

| Model | Precision | Recall | Hit Rate | MRR |
|-------|----------:|-------:|---------:|----:|
| Qwen RAG | 0.2827 | 0.2895 | 0.7323 | 0.5268 |
| Graph RAG | 0.2341 | 0.2298 | 0.6850 | 0.4433 |
| CamELBERT RAG | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Changes vs. Previous Run

| Fix | Impact |
|-----|--------|
| Broad `\s*\([^)]*\)` regex strips Graph RAG's `(graph_score=..., pagerank=...)` annotations | Graph RAG sources now parsed correctly |
| Normalized question lookup (NFKC + diacritic strip) | Questions matched even with minor encoding differences |
| Deduplication of retrieved sources before metric computation | Precision no longer deflated by duplicate retrievals |
| Unicode NFKC + Arabic diacritic normalization on both sides | Robust cross-encoding source comparison |