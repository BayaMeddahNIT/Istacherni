# dense_rag/bge_retriever.py
"""
Semantic retrieval using BGE-M3 + ChromaDB.
Drop-in companion to bm25_retriever.py.
"""

import chromadb
from pathlib import Path
from dense_rag.bge_embedder import embed_query

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "algerian_law"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def dense_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """
    Retrieve top-K articles by semantic similarity.
    Returns same schema as bm25_retrieve() for easy fusion.
    """
    collection = _get_collection()
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    
    retrieved = []
    for meta, distance in zip(
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Cosine distance → similarity score (1 = identical)
        score = round(1 - distance, 4)
        retrieved.append({
            **meta,
            "score": score,
            "retrieval_method": "dense",
        })
    
    return retrieved