# qwen_rag — Dense Vector RAG for Algerian Law

Local dense retrieval pipeline using **Qwen3-Embedding-8B** (HuggingFace) and
**ChromaDB**, with generation via **Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled** running locally through **Ollama**.

Completely independent from `bm25_rag/`.

---

## Architecture

```
Question
   │
   ▼
qwen_embedder.py          ← Qwen3-Embedding-8B (local, transformers)
   │  embed_query()           wraps query in instruction prefix
   │
   ▼
qwen_retriever.py         ← ChromaDB cosine search
   │  qwen_retrieve()         returns top-K article dicts
   │
   ▼
qwen_generator.py         ← Ollama  (Qwen3.5-27B)
   │  qwen_generate()         builds Arabic legal answer
   │
   ▼
Arabic Answer
```

---

## First-run setup

### 1 — Install Python dependencies

```bash
pip install transformers torch sentence-transformers chromadb python-dotenv
```

> For GPU acceleration install the CUDA-enabled torch wheel from https://pytorch.org.

### 2 — Build the ChromaDB index (run once)

```bash
python qwen_rag/qwen_indexer.py
```

This loads the dataset, embeds all articles with Qwen3-Embedding-8B,
and writes a persistent ChromaDB store to `qwen_rag/chroma_db/`.

### 3 — Install and start Ollama

```bash
# Install from https://ollama.com/download, then:
ollama serve                   # keep running in a terminal
ollama pull qwen3.5:27b-claude-4.6-opus-reasoning-distilled
```

### 4 — Test the full pipeline

```bash
python qwen_rag/qwen_pipeline.py
```

---

## Usage in code

```python
from qwen_rag.qwen_pipeline import qwen_ask

# Simple
answer = qwen_ask("ما هي عقوبة غش المواد الغذائية؟")
print(answer)

# With sources
answer, sources = qwen_ask(
    "ما هي شروط عقد البيع؟",
    top_k=5,
    return_sources=True,
)

# Retrieval only
from qwen_rag.qwen_retriever import qwen_retrieve
hits = qwen_retrieve("حقوق العامل عند الفصل", top_k=3)

# Generation only
from qwen_rag.qwen_generator import qwen_generate
text = qwen_generate(question, hits)
```

---

## Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `QWEN_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | HF model ID |
| `QWEN_EMBED_BATCH_SIZE` | `8` | Embedding batch size |
| `QWEN_EMBED_MAX_SEQ_LEN` | `512` | Token truncation limit |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3.5:27b-claude-4.6-opus-reasoning-distilled` | Ollama model tag |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (s) |
| `OLLAMA_NUM_CTX` | `4096` | Context window |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature |

---

## Module overview

| File | Role |
|---|---|
| `qwen_loader.py` | Dataset loader (delegates to `bm25_rag.bm25_loader`) |
| `qwen_embedder.py` | Qwen3-Embedding-8B wrapper with instruction prompting |
| `qwen_indexer.py` | Builds / loads the ChromaDB collection |
| `qwen_retriever.py` | Cosine-similarity retrieval |
| `qwen_generator.py` | Ollama generation + health check |
| `qwen_pipeline.py` | End-to-end entry point + interactive CLI |
