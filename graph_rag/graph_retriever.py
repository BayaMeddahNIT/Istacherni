"""
graph_retriever.py
------------------
Retrieval via graph traversal — the key distinction from Dense/BM25 RAG.

Algorithm (3 hops, explained visually):

  Query: "ما هي عقوبة السرقة؟"
  Extracted terms: ["سرقة", "عقوبة"]

  Step 1 — SEED (hop 0)
    Find concept/domain/penalty nodes whose label contains any query term.
    → CONCEPT:سرقة  CONCEPT:اختلاس  PENALTY:الحبس …

  Step 2 — HOP 1
    From those seed nodes, walk edges BACKWARD to find article nodes.
    → DZ_PENAL_ART_350, DZ_PENAL_ART_351, DZ_PENAL_ART_372 …

  Step 3 — HOP 2
    From the article nodes found above, follow RELATED_TO edges to
    collect their structurally linked articles.
    → DZ_PENAL_ART_14 (referenced by 350/351) …

  Step 4 — HOP 3 (SAME_LAW broadening)
    Optionally add up to N siblings from the same law for extra context.

  Step 5 — SCORE & RANK
    Rank the discovered articles by:
      - graph_score  = number of times an article was reached via the graph
      - degree_score = PageRank-like importance (pre-computed on the full graph)
    Return top_k articles with all metadata.
"""

import pickle
import re
from pathlib import Path
from typing import Optional

import networkx as nx

from graph_rag.graph_builder import load_graph, CACHE_DIR

# ── Lazy singletons ────────────────────────────────────────────────────────────
_G:      Optional[nx.DiGraph] = None
_corpus: Optional[list]       = None
_PR:     Optional[dict]       = None   # pre-computed PageRank

_PR_CACHE = CACHE_DIR / "pagerank.pkl"


def _get_graph() -> tuple[nx.DiGraph, list, dict]:
    global _G, _corpus, _PR
    if _G is None:
        _G, _corpus = load_graph()
        if _PR_CACHE.exists():
            print("[GraphRetriever] Loading PageRank from cache…")
            with open(_PR_CACHE, "rb") as f:
                _PR = pickle.load(f)
        else:
            print("[GraphRetriever] Computing PageRank (first time — will be cached)…")
            _PR = nx.pagerank(_G, alpha=0.85, max_iter=200)
            with open(_PR_CACHE, "wb") as f:
                pickle.dump(_PR, f)
            print("[GraphRetriever] PageRank cached.")
        print(f"[GraphRetriever] Ready — {_G.number_of_nodes()} nodes.")
    return _G, _corpus, _PR


# ═══════════════════════════════════════════════════════════════════════
# ── Query-term extractor ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _normalize_term(t: str) -> str:
    """Strip diacritics + normalise common Arabic letter variants."""
    t = re.sub(r"[\u064B-\u065F\u0640]", "", t)
    t = re.sub(r"[أإآا]", "ا", t)
    t = re.sub(r"ة",       "ه", t)
    t = re.sub(r"ى",       "ي", t)
    return t.strip()


def _extract_query_terms(query: str) -> list[str]:
    """Tokenise and normalise Arabic query into searchable terms (≥ 2 chars)."""
    cleaned = re.sub(r"[\u064B-\u065F\u0640]", "", query)
    cleaned = re.sub(r"[أإآا]", "ا", cleaned)
    cleaned = re.sub(r"ة",       "ه", cleaned)
    cleaned = re.sub(r"ى",       "ي", cleaned)
    tokens = re.split(r"[^\w\u0600-\u06FF]+", cleaned.lower())
    return [t for t in tokens if len(t) >= 2]


# ── Stop-words to skip when matching concept nodes ────────────────────
_STOPWORDS = {
    "في", "من", "على", "ان", "الى", "هي", "ما", "هل", "كيف",
    "له", "لها", "عن", "مع", "حول", "اذا", "اذ", "التي", "الذي",
    "هذا", "هذه", "تلك", "ذلك", "بعد", "قبل", "خلال",
}


# ═══════════════════════════════════════════════════════════════════════
# ── Core retrieval ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def graph_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the top-K most relevant law articles via Knowledge Graph traversal.

    Args:
        query:  User's legal question (Arabic or French).
        top_k:  Number of articles to return.

    Returns:
        List of article dicts sorted by graph relevance score, each with an
        added "graph_score" and "pagerank" key.
    """
    G, corpus, PR = _get_graph()

    query_terms = [
        _normalize_term(t)
        for t in _extract_query_terms(query)
        if t not in _STOPWORDS
    ]
    if not query_terms:
        return []

    # ── Step 1: Find SEED nodes ───────────────────────────────────────
    seed_nodes: set[str] = set()
    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "")

        if node_type == "concept":
            term_norm = _normalize_term(data.get("term", ""))
            for qt in query_terms:
                if qt in term_norm or term_norm in qt:
                    seed_nodes.add(node)

        elif node_type == "penalty":
            p_norm = _normalize_term(data.get("penalty_class", ""))
            for qt in query_terms:
                if qt in p_norm or p_norm in qt:
                    seed_nodes.add(node)

        elif node_type == "domain":
            d_norm = _normalize_term(data.get("name", ""))
            for qt in query_terms:
                if qt in d_norm:
                    seed_nodes.add(node)

    # ── Step 2: HOP 1 — articles directly linked to seed nodes ───────
    hop1_articles: dict[str, int] = {}   # article_id → hit count
    for seed in seed_nodes:
        # Predecessors (nodes that point TO the seed via edges like HAS_KEYWORD)
        for pred in G.predecessors(seed):
            if G.nodes[pred].get("node_type") == "article":
                hop1_articles[pred] = hop1_articles.get(pred, 0) + 1
        # Successors too (for CONCEPT → ARTICLE edges if any)
        for succ in G.successors(seed):
            if G.nodes[succ].get("node_type") == "article":
                hop1_articles[succ] = hop1_articles.get(succ, 0) + 1

    # Also: if any query term appears in an article node's own keywords / title
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "article":
            continue
        art_keywords_norm = [_normalize_term(k) for k in data.get("keywords", [])]
        title_norm        = _normalize_term(data.get("title", ""))
        hits = sum(
            1 for qt in query_terms
            if any(qt in kn or kn in qt for kn in art_keywords_norm)
            or qt in title_norm
        )
        if hits > 0:
            hop1_articles[node] = hop1_articles.get(node, 0) + hits

    if not hop1_articles:
        # Fallback: return the highest-PR article nodes (global importance)
        art_nodes = [(n, PR[n]) for n, d in G.nodes(data=True) if d.get("node_type") == "article"]
        art_nodes.sort(key=lambda x: x[1], reverse=True)
        result = []
        for node_id, pr_score in art_nodes[:top_k]:
            d = G.nodes[node_id]
            result.append(_node_to_result(node_id, d, graph_score=0, pagerank=pr_score))
        return result

    # ── Step 3: HOP 2 — RELATED_TO neighbours of hop-1 articles ──────
    hop2_articles: dict[str, int] = {}
    for art_id in hop1_articles:
        for succ in G.successors(art_id):
            if (G.nodes[succ].get("node_type") == "article"
                    and succ not in hop1_articles):
                rel = G.edges[art_id, succ].get("rel", "")
                if rel == "RELATED_TO":
                    hop2_articles[succ] = hop2_articles.get(succ, 0) + 1

    # ── Step 4: Merge and score ────────────────────────────────────────
    all_candidates: dict[str, float] = {}
    for art_id, hits in hop1_articles.items():
        # hop-1 articles get full weight + PageRank bonus
        all_candidates[art_id] = hits * 2.0 + PR.get(art_id, 0) * 10

    for art_id, hits in hop2_articles.items():
        # hop-2 articles get half weight
        all_candidates[art_id] = hits * 1.0 + PR.get(art_id, 0) * 10

    # Sort by combined score
    ranked = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)

    # ── Step 5: Build result dicts ────────────────────────────────────
    results = []
    for art_id, score in ranked[:top_k]:
        d = G.nodes.get(art_id, {})
        if not d:
            continue
        results.append(_node_to_result(art_id, d, graph_score=round(score, 4),
                                       pagerank=round(PR.get(art_id, 0), 6)))
    return results


def _node_to_result(node_id: str, data: dict, graph_score: float, pagerank: float) -> dict:
    return {
        "id":                         node_id,
        "law_name":                   data.get("law_name", ""),
        "law_domain":                 data.get("law_domain", ""),
        "article_number":             data.get("article_number", ""),
        "title":                      data.get("title", ""),
        "text_original":              data.get("text_original", ""),
        "penalties_summary":          data.get("penalties_summary", ""),
        "legal_conditions_summary":   data.get("legal_conditions_summary", ""),
        "keywords":                   data.get("keywords", []),
        "graph_score":                graph_score,
        "pagerank":                   pagerank,
    }


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "ما هي عقوبة السرقة في القانون الجزائري؟",
        "ما هي شروط عقد البيع؟",
        "ما هي حقوق العامل عند الفصل التعسفي؟",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query : {q}")
        results = graph_retrieve(q, top_k=5)
        print(f"Found : {len(results)} results")
        for r in results:
            print(f"  [{r['law_name']}] مادة {r['article_number']:6s}  "
                  f"graph={r['graph_score']:.3f}  pr={r['pagerank']:.6f}  "
                  f"| {r['title'][:50]}")
