"""
minilm_rag/
-----------
Hybrid RAG using:
  - BM25 (keyword matching)  via rank_bm25
  - Dense embeddings          via sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - Generation                via Gemini 2.5 Flash

Public API:
  from minilm_rag.minilm_retriever import minilm_retrieve
  from minilm_rag.minilm_generator import minilm_generate
"""
