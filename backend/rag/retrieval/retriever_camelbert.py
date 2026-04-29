"""
camelbert_retriever.py
-----------------------
Retrieval module: embeds a user query with the same local CamELBERT model
used during ingestion and finds the top-K most relevant law article chunks
from the CamELBERT-specific ChromaDB collection.

Drop-in replacement for retriever.py — same retrieve() signature:
    retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]

Usage (standalone test):
  cd <project root>
  python backend/rag/retrieval/camelbert_retriever.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import chromadb
from transformers import AutoTokenizer, AutoModel

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.preprocessing.normalize_arabic import normalize_arabic

# ── Configuration — must match camelbert_embed_articles.py ────────────────────
CAMELBERT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
COLLECTION_NAME = "algerian_law_camelbert"
VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore_camelbert"
DEFAULT_TOP_K   = 5
MAX_SEQ_LEN     = 512

# ── Device selection ───────────────────────────────────────────────────────────
DEVICE = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)

# ── Lazy singletons ────────────────────────────────────────────────────────────
_tokenizer        = None
_model            = None
_chroma_collection = None


def _get_model():
    """Load CamELBERT tokenizer + model once, then reuse."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(CAMELBERT_MODEL)
        _model     = AutoModel.from_pretrained(CAMELBERT_MODEL).to(DEVICE)
        _model.eval()
    return _tokenizer, _model


def _get_collection():
    """Load the persistent CamELBERT ChromaDB collection (lazy singleton)."""
    global _chroma_collection
    if _chroma_collection is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"CamELBERT vector store not found at {VECTORSTORE_DIR}.\n"
                "Run `python backend/rag/embedding/camelbert_embed_articles.py` first."
            )
        client             = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


# ── Embedding ──────────────────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask       = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string using CamELBERT.
    Returns an L2-normalised mean-pool vector as a plain Python float list.
    """
    tokenizer, model = _get_model()

    encoded = tokenizer(
        [query],                  # tokenizer expects a list
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)

    pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled[0].cpu().tolist()   # single vector → flat list


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant law articles for the given query.

    The query is normalised (diacritics removed, letter variants unified) then
    embedded with CamELBERT before being searched against the ChromaDB index.

    Args:
        query:  Natural-language question in Arabic (or French — Arabic preferred).
        top_k:  Number of results to return (default 5).

    Returns:
        List of dicts with keys: id, text, metadata, score
        'score' is the cosine distance returned by ChromaDB (lower = more similar).
    """
    # Normalise query the same way the documents were normalised at index time
    normalised_query  = normalize_arabic(query)
    query_embedding   = embed_query(normalised_query)

    collection = _get_collection()
    n          = min(top_k, collection.count())

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = n,
        include          = ["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc_id, doc_text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "id"      : doc_id,
            "text"    : doc_text,
            "metadata": meta,
            "score"   : round(dist, 4),
        })

    return retrieved


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Query : {test_query}\n")
    results = retrieve(test_query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"Result {i}  [score={r['score']}]")
        print(f"  ID     : {r['id']}")
        print(f"  Law    : {r['metadata'].get('law_name', '')}")
        print(f"  Article: {r['metadata'].get('article_number', '')}")
        print(f"  Text   : {r['metadata'].get('text_original', r['text'])[:200]}...")
        print()