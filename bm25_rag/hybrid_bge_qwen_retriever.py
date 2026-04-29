"""
hybrid_bge_qwen_retriever.py
----------------------------
Hybrid retrieval: combines BM25 sparse retrieval with BAAI/BGE-M3
dense (ChromaDB) retrieval using Reciprocal Rank Fusion (RRF).

Pipeline:
  1. BM25 retrieval   → top-K ranked list  (lexical match)
  2. BGE-M3 retrieval → top-K ranked list  (semantic match, ChromaDB)
  3. RRF fusion       → merged top-K list  (best of both worlds)

NOTE: BGE-M3 query embedding uses sentence-transformers (not FlagEmbedding)
because FlagEmbedding has a transformers version conflict in this environment.
The ChromaDB collection was built with FlagEmbedding (1024-dim dense vectors);
sentence-transformers loads the same BAAI/bge-m3 weights and produces
identical embeddings — fully compatible with the existing collection.

# ── OLD: FlagEmbedding import (kept as comment) ──────────────────────────────
# from rag_bge.bge_embedder import embed_query as bge_embed_query
# This used FlagEmbedding.BGEM3FlagModel which has a transformers conflict:
#   ImportError: cannot import name 'is_torch_fx_available'
# ──────────────────────────────────────────────────────────────────────────────

Usage (standalone):
  cd <project root>
  python bm25_rag/hybrid_bge_qwen_retriever.py

Usage (in pipeline):
  from bm25_rag.hybrid_bge_qwen_retriever import hybrid_retrieve
  results = hybrid_retrieve("ما هي عقوبة السرقة؟", top_k=5)
"""

import sys
import os
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from bm25_rag.bm25_retriever import bm25_retrieve





# ── Configuration ──────────────────────────────────────────────────────────────
BGE_MODEL_NAME      = "BAAI/bge-m3"
BGE_COLLECTION_NAME = "algerian_law_bge"
BGE_VECTORSTORE_DIR = PROJECT_ROOT / "backend" / "vectorstore"
RRF_K               = 60          # standard RRF constant
DEFAULT_TOP_K       = 5

# ── Lazy singletons ────────────────────────────────────────────────────────────
_bge_collection:  Optional[chromadb.Collection] = None
_bge_available:   Optional[bool]                = None

def _get_bge_embedding(text: str) -> List[float]:
    """Get embeddings from Ollama instead of local sentence-transformers."""
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/embeddings"
    payload = json.dumps({
        "model": "bge-m3",
        "prompt": text
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url, 
        data=payload, 
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]

    except Exception as e:
        print(f"[HybridRetriever] Ollama Embedding Error: {e}")
        # Fallback or re-raise
        raise RuntimeError(f"Could not get embeddings from Ollama: {e}")




def _get_bge_collection() -> chromadb.Collection:
    """Open the persistent ChromaDB collection (cached after first call)."""
    global _bge_collection
    if _bge_collection is None:
        if not BGE_VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"BGE vector store not found at {BGE_VECTORSTORE_DIR}.\n"
                "Run `python rag_bge/bge_embed_articles.py` first."
            )
        client = chromadb.PersistentClient(path=str(BGE_VECTORSTORE_DIR))
        _bge_collection = client.get_collection(name=BGE_COLLECTION_NAME)
    return _bge_collection


def _bge_retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Dense retrieval using BAAI/BGE-M3 embeddings stored in ChromaDB.

    Uses sentence-transformers to embed the query (same model weights as
    FlagEmbedding, fully compatible with the existing ChromaDB collection).
    """
    # Use Ollama for embedding instead of crashing SentenceTransformer
    vector = _get_bge_embedding(query)


    collection = _get_bge_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc_id, doc_text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "id":                        doc_id,
            "law_name":                  meta.get("law_name", ""),
            "law_domain":                meta.get("law_domain", ""),
            "article_number":            meta.get("article_number", ""),
            "title":                     meta.get("title", ""),
            "text_original":             meta.get("text_original", doc_text),
            "keywords":                  [],
            "penalties_summary":         meta.get("penalties_summary", ""),
            "legal_conditions_summary":  meta.get("legal_conditions_summary", ""),
            "score":                     round(float(dist), 4),
        })
    return retrieved


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_merge(
    bm25_results: List[Dict],
    bge_results:  List[Dict],
    top_k: int,
    rrf_k: int = RRF_K,
) -> List[Dict]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.
    rrf_score(d) = Σ  1 / (rrf_k + rank_i(d))
    """
    scores:    Dict[str, float] = {}
    best_meta: Dict[str, Dict]  = {}

    def _add(ranked: List[Dict]):
        for rank, item in enumerate(ranked, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in best_meta:
                best_meta[doc_id] = item

    _add(bm25_results)
    _add(bge_results)

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    merged = []
    for doc_id in sorted_ids[:top_k]:
        item = best_meta[doc_id].copy()
        item["score"] = round(scores[doc_id], 6)
        merged.append(item)
    return merged


# ── Public API ─────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    bm25_candidates: int = 20,
    bge_candidates: int  = 20,
) -> List[Dict[str, Any]]:
    """
    Hybrid BM25 + BGE-M3 retrieval with RRF fusion.

    Falls back to BM25-only if the BGE-M3 collection or model is unavailable.

    Args:
        query:           User's legal question (Arabic / French / English).
        top_k:           Final number of results to return.
        bm25_candidates: Candidates from BM25 before fusion.
        bge_candidates:  Candidates from BGE-M3 before fusion.

    Returns:
        List of article dicts ranked by RRF (or BM25) score, highest first.
    """
    global _bge_available

    # ── Step 1: BM25 retrieval (always available) ─────────────────────────────
    bm25_results = bm25_retrieve(query, top_k=bm25_candidates)

    # ── Step 2: BGE-M3 dense retrieval (with fallback) ───────────────────────
    bge_results: List[Dict] = []
    if _bge_available is not False:          # skip if previously failed
        try:
            bge_results = _bge_retrieve(query, top_k=bge_candidates)
            _bge_available = True
        except Exception as e:
            _bge_available = False
            print(f"[HybridRetriever] BGE-M3 unavailable, using BM25-only: {e}")

    # ── Step 3: Fuse or return BM25-only ─────────────────────────────────────
    if bge_results:
        return _rrf_merge(bm25_results, bge_results, top_k=top_k)
    else:
        # BM25-only fallback: clip to top_k
        return bm25_results[:top_k]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
    ]
    for q in queries:
        print(f"\n{'='*65}")
        print(f"Query: {q}")
        print(f"{'='*65}")
        results = hybrid_retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['law_name']} — المادة {r['article_number']}")
            print(f"       Score : {r['score']}")
            print(f"       Text  : {r['text_original'][:150]}…")
