"""
graph_retriever.py
------------------
Retrieval via graph traversal ENHANCED with BGE-M3 local embeddings.
Instead of just keyword matching, we use vector similarity to find seed nodes.
"""

import pickle
import re
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

# Import local utilities
from graph_rag_local.embeddings import embed_text, cosine_similarity
from graph_rag_local.graph_builder import GRAPH_FILE, CORPUS_FILE, CACHE_DIR
from graph_rag_local.query_classifier import classify_query

# ── Domain Name Normalization ──────────────────────────────────────────────────
# CamelBERT returns English domain names; the dataset stores a slightly different
# naming convention. This map normalises them so the domain filter actually works.
# Add new entries here if you add new law files with different law_domain strings.
DOMAIN_MAP: dict[str, str] = {
    "Civil Law":                                   "Civil Law",
    "Penal Code":                                  "Penal Law",
    "Penal Law":                                   "Penal Law",
    "Labor Law":                                   "Labor Law",
    "Commercial Law":                              "Commercial Law",
    "Code of Civil and Administrative Procedures": "Civil and Administrative Procedures",
}

# ── Lazy singletons ────────────────────────────────────────────────────────────
_G:      Optional[nx.DiGraph] = None
_corpus: Optional[list]       = None
_PR:     Optional[dict]       = None   # PageRank
_node_embs: Optional[np.ndarray] = None # Matrix of article embeddings
_node_ids:   Optional[list]      = None
_bm25:       Optional[BM25Okapi] = None

_PR_CACHE = CACHE_DIR / "pagerank_local.pkl"

def _tokenize(text: str) -> list[str]:
    """Strip diacritics, normalise Arabic letter variants, split into tokens."""
    if not text: return []
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ة",       "ه", text)
    text = re.sub(r"ى",       "ي", text)
    tokens = re.split(r"[^\w\u0600-\u06FF]+", text.lower())
    return [t for t in tokens if len(t) >= 2]

def _get_resources() -> tuple[nx.DiGraph, list, dict, np.ndarray, list, BM25Okapi]:
    global _G, _corpus, _PR, _node_embs, _node_ids, _bm25
    if _G is None:
        if not GRAPH_FILE.exists():
            raise FileNotFoundError(f"Graph cache not found. Run python -m graph_rag_local.graph_builder first.")
        
        with open(GRAPH_FILE, "rb") as f: _G = pickle.load(f)
        with open(CORPUS_FILE, "rb") as f: _corpus = pickle.load(f)
        
        if _PR_CACHE.exists():
            with open(_PR_CACHE, "rb") as f: _PR = pickle.load(f)
        else:
            print("[GraphRetriever] Computing PageRank...")
            _PR = nx.pagerank(_G, alpha=0.85)
            with open(_PR_CACHE, "wb") as f: pickle.dump(_PR, f)

        # Build embedding matrix for fast similarity search
        print("[GraphRetriever] Pre-loading embedding matrix...")
        article_nodes = [(n, d["embedding"]) for n, d in _G.nodes(data=True) if d.get("node_type") == "article" and "embedding" in d]
        if article_nodes:
            _node_ids, embs = zip(*article_nodes)
            _node_embs = np.array(embs)
            
            print("[GraphRetriever] Building BM25 index...")
            tokenized_corpus = []
            for n_id in _node_ids:
                d = _G.nodes[n_id]
                text_to_tokenize = f"{d.get('title', '')} {d.get('text_original', '')} {d.get('summary', '')} {' '.join(d.get('keywords', []))}"
                tokenized_corpus.append(_tokenize(text_to_tokenize))
            _bm25 = BM25Okapi(tokenized_corpus)
        else:
            _node_ids, _node_embs, _bm25 = [], np.array([]), None
            
    return _G, _corpus, _PR, _node_embs, _node_ids, _bm25


def expand_acronyms(query: str) -> str:
    """
    Replaces common Algerian French legal acronyms with their formal MSA Arabic
    equivalents using word-boundary matching. This prevents the LLM from ever
    seeing the French letters and hallucinating French translations.
    e.g. "ما الفرق بين SARL و SPA؟" →
         "ما الفرق بين شركة ذات مسؤولية محدودة و شركة مساهمة؟"
    """
    acronym_map = {
        "SARL": "شركة ذات مسؤولية محدودة",
        "EURL": "مؤسسة ذات شخص وحيد وذات مسؤولية محدودة",
        "SPA":  "شركة مساهمة",
        "SNC":  "شركة التضامن",
        "SCA":  "شركة التوصية بالأسهم",
        "SCS":  "شركة التوصية البسيطة",
    }

    expanded_query = query
    for acronym, arabic_term in acronym_map.items():
        # \b gives us word boundaries so "SPA" doesn't match inside "SPAIN" etc.
        # re.IGNORECASE handles lowercase variants like "sarl"
        expanded_query = re.sub(
            rf"\b{re.escape(acronym)}\b",
            arabic_term,
            expanded_query,
            flags=re.IGNORECASE
        )

    if expanded_query != query:
        print(f"[Acronym Expander] '{query}' → '{expanded_query}'")

    return expanded_query


# ── Legal Vocabulary Normalizer ────────────────────────────────────────────────
# Maps everyday colloquial Arabic terms → official Algerian legal terminology.
# This bridges the gap between how citizens speak and how the law is written,
# enabling bge-m3 to match against the correct indexed vocabulary.
# Add entries here whenever you observe a vocabulary mismatch in retrieval logs.
_LEGAL_SYNONYMS: dict[str, str] = {
    # Dismissal / Termination (قانون العمل)
    "فصل من العمل":    "التسريح التعسفي",
    "الفصل من العمل":  "التسريح التعسفي",
    "طردت من العمل":   "التسريح التعسفي",
    "طرد من العمل":    "التسريح التعسفي",
    "رفت من العمل":    "التسريح التعسفي",
    "طردني صاحب العمل":"التسريح التأديبي التعسفي",
    "فصلني المدير":    "التسريح التأديبي",
    "فقدت عملي":       "إنهاء عقد العمل",

    # Wages / Salary
    "مرتبي":           "الأجر",
    "راتبي":           "الأجر",
    "ما دفعولي":       "عدم دفع الأجر",
    "ما خلصوني":       "عدم دفع الأجر",

    # Contract
    "عقدي":            "عقد العمل",
    "عندي عقد":        "عقد العمل",

    # Business registration
    "فتح شركة":        "تأسيس الشركة السجل التجاري",
    "نفتح شركة":       "تأسيس الشركة",

    # Theft / Crime (قانون العقوبات)
    "سرقني":           "جريمة السرقة",
    "ضربني":           "جريمة الضرب والجرح العمد",
    "شتمني":           "جريمة القذف والسب",

    # Family / Personal Status
    "يطلقها":          "الطلاق",
    "خلع":             "الطلاق بالخلع",
    "مسكن الزوجية":    "السكن الزوجي حق السكن",

    # Fix E: Cross-domain criminal fraud vocabulary.
    # These are commerce-framed queries whose CORRECT legal domain is قانون العقوبات.
    # Without this, "نصب في التجارة" anchors to القانون التجاري, missing م.429.
    "نصب في التجارة":  "احتيال خداع تدليس قانون العقوبات",
    "نصب التجاري":     "احتيال خداع تدليس قانون العقوبات",
    "نصب":             "احتيال تدليس خداع",
    "خدعني التاجر":    "احتيال تدليس غش تجاري قانون العقوبات",
    "خدعني":           "احتيال تدليس غش",
    "بضاعة مزورة":     "غش تجاري تقليد بضاعة قانون العقوبات",
    "منتج مزور":       "غش تجاري تقليد بضاعة قانون العقوبات",
    "سلعة مزورة":      "غش تجاري تقليد بضاعة قانون العقوبات",
    "بيع منتج مغشوش":  "غش بيع السلع خداع المستهلك قانون العقوبات",
    "لم يسلم السلعة":  "عدم تسليم المبيع احتيال اختلاس قانون العقوبات",
    "عدم تسليم السلعة":"عدم تسليم المبيع احتيال اختلاس قانون العقوبات",
    "ما سلملي":        "عدم تسليم المبيع احتيال اختلاس",
    "تزوير فاتورة":    "تزوير محررات عقوبات",
    "تزوير فواتير":    "تزوير محررات عقوبات",
    "فاتورة مزورة":    "تزوير محررات وثائق قانون العقوبات",
}

def normalize_legal_slang(query: str) -> str:
    """
    Replaces colloquial Arabic expressions with their official legal equivalents.
    Uses longest-match-first to avoid partial replacements.
    """
    normalized = query
    # Sort by length descending so longer phrases match before substrings
    for slang, legal_term in sorted(_LEGAL_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if slang in normalized:
            normalized = normalized.replace(slang, legal_term)
            print(f"[LegalNorm] '{slang}' → '{legal_term}'")
    return normalized



def graph_retrieve(query: str, top_k: int = 12, vector_top_n: int = 80, seeds: list[str] = None) -> list[dict]:
    """
    Retrieve top-K law articles using Hybrid Seed Discovery:
    1. Hybrid Search (Qwen3-Embedding + BM25) to find semantically and lexically relevant seed articles.
    2. Graph Traversal (Hops) to find structurally linked articles.
    3. Final Ranking by combined Score + PageRank.
    """
    G, corpus, PR, node_embs, node_ids, bm25_index = _get_resources()
    if len(node_ids) == 0: return []
    seeds = seeds or []

    # ── Pre-Retrieval: Intent Classification ───────────────────────
    expanded_query = expand_acronyms(query)
    expanded_query = normalize_legal_slang(expanded_query)   # colloquial → legal vocab
    print(f"\n[GraphRetriever] Classifying query: {expanded_query}")
    intent = classify_query(expanded_query)
    amplified_query = intent.get("amplified_query", expanded_query)
    hyde_doc = intent.get("hyde_document", "")
    synonyms = intent.get("synonyms", [])
    predicted_domain = intent.get("domain", "")
    target_keywords = set(intent.get("keywords", []))
    negative_concepts = set(intent.get("negative_concepts", []))
    
    print(f"[GraphRetriever] Amplified: {amplified_query}")
    if hyde_doc:
        print(f"[GraphRetriever] HyDE Generated: {hyde_doc[:100]}...")
    if synonyms:
        print(f"[GraphRetriever] Synonyms: {', '.join(synonyms)}")
        
    if predicted_domain:
        # Normalise to match the law_domain strings stored in the graph nodes
        predicted_domain = DOMAIN_MAP.get(predicted_domain, predicted_domain)
        print(f"[GraphRetriever] Domain focus: {predicted_domain}")

    # ── Step 1: Hybrid SEED Discovery (Vector + BM25) ───────────────────────────
    # Inject extracted seeds directly into the search string so BM25 forces them to the top
    search_text = f"{amplified_query} {hyde_doc} {' '.join(synonyms)} {' '.join(seeds)}"
    query_emb = embed_text(search_text)
    similarities = cosine_similarity(query_emb, node_embs)
    
    tokenized_query = _tokenize(search_text)
    bm25_scores = bm25_index.get_scores(tokenized_query) if bm25_index and tokenized_query else np.zeros(len(node_ids))
    
    # Reciprocal Rank Fusion (RRF)
    k_rrf = 60
    
    # Rank semantic similarities
    dense_ranks = {node_ids[idx]: rank + 1 for rank, idx in enumerate(np.argsort(similarities)[::-1])}
    
    # Rank BM25 scores
    sparse_ranks = {node_ids[idx]: rank + 1 for rank, idx in enumerate(np.argsort(bm25_scores)[::-1])}
    
    rrf_scores = {}
    for n_id in node_ids:
        dense_rank = dense_ranks.get(n_id, len(node_ids))
        sparse_rank = sparse_ranks.get(n_id, len(node_ids))
        score = 1.0 / (k_rrf + dense_rank) + 1.0 / (k_rrf + sparse_rank)
        rrf_scores[n_id] = score
        
    # Get top-N seeds from RRF scores
    top_n_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:vector_top_n]
    vector_seeds = {n_id: rrf_scores[n_id] for n_id in top_n_ids}

    # ── Step 2: Advanced Concept Checking & Domain Boost ───────────────
    hop1_articles: dict[str, float] = {}   # article_id → score
    
    for seed_id, sim in vector_seeds.items():
        node_data = G.nodes[seed_id]
        
        # Domain soft-filtering: penalize out-of-domain, BOOST in-domain
        node_domain = node_data.get("law_domain", "")
        if predicted_domain and node_domain:
            if predicted_domain == node_domain or predicted_domain in node_domain or node_domain in predicted_domain:
                sim *= 2.0  # Strong boost for confirmed in-domain articles
            else:
                sim *= 0.3  # Stronger penalty for out-of-domain
            
        # Taxonomy hard-filtering (Substantive vs Procedural)
        target_law_type = intent.get("law_type", "both")
        node_law_type = node_data.get("law_type", "both")
        if target_law_type in ["substantive", "procedural"] and node_law_type in ["substantive", "procedural"]:
            if target_law_type != node_law_type:
                sim *= 0.05  # Massively penalize cross-contamination (e.g. proof vs validity)
            
        # Negative concepts filtering (e.g., separating "promise to contract" from "contract validity")
        node_title = node_data.get("title", "")
        node_keywords = set(node_data.get("keywords", []))
        
        is_negative = any(neg in node_title for neg in negative_concepts) or bool(node_keywords & negative_concepts)
        if is_negative:
            sim *= 0.1  # Massively penalize negative matches
            
        # Concept boosting
        if bool(node_keywords & target_keywords):
            sim *= 1.5  # Boost exact legal concept matches
        
        hop1_articles[seed_id] = hop1_articles.get(seed_id, 0) + sim * 2.0
        
        # (Edges are no longer explicitly followed here; Personalized PageRank handles topology)

    # ── Step 3: Personalized PageRank (PPR) ─────────────────────────
    # Dynamically traverse the graph based on the initial retrieved nodes
    top_candidates = sorted(hop1_articles.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if top_candidates:
        personalization = {seed_id: 1.0 / len(top_candidates) for seed_id, _ in top_candidates}
        # Run PPR with dynamic alpha based on whether explicit seeds were provided
        alpha_val = 0.90 if len(seeds) > 0 else 0.70
        ppr_scores = nx.pagerank(G, alpha=alpha_val, personalization=personalization, max_iter=50)
        
        # Extract top 2 nodes with highest PPR that were not heavily retrieved
        ppr_ranked = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        added_ppr = 0
        
        for n_id, ppr_score in ppr_ranked:
            if G.nodes[n_id].get("node_type") != "article": continue
            if n_id not in hop1_articles and added_ppr < 2:
                # Semantic Gate: Ensure the PPR node is actually contextually relevant
                try:
                    n_idx = node_ids.index(n_id)
                    sim = float(cosine_similarity(query_emb, node_embs[n_idx:n_idx+1])[0])
                except ValueError:
                    sim = 0.0
                    
                if sim < 0.25:
                        print(f"[GraphRetriever] PPR node {n_id} rejected (Semantic Gate: sim={sim:.3f} < 0.25)")
                        continue
                
                # Add PPR-discovered nodes (subgraph context) with a 0.0 baseline RRF
                hop1_articles[n_id] = 0.0
                added_ppr += 1
                print(f"[GraphRetriever] PPR subgraph pulled in: {n_id} (sim={sim:.3f})")

        # Rebalance Scoring Weights: final_node_score = (0.85 * rrf_score) + (0.15 * ppr_score)
        # ppr_score is very small (e.g., 0.01-0.1). We scale it by 50 to match RRF magnitudes.
        for art_id in hop1_articles:
            rrf_score = hop1_articles[art_id]
            scaled_ppr = ppr_scores.get(art_id, 0) * 50
            hop1_articles[art_id] = (0.85 * rrf_score) + (0.15 * scaled_ppr)

    # ── Step 4: Final Rank and Format ──────────────────────────────────────
    final_scores = {}
    for art_id, score in hop1_articles.items():
        # Global PR is reduced, keeping PPR & RRF dominant
        final_scores[art_id] = score + PR.get(art_id, 0) * 2
        
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # USE_CROSS_ENCODER = False  # Future GPU upgrade stub
    # if USE_CROSS_ENCODER and ranked:
    #     # Re-rank the top 15 results through a Cross-Encoder (e.g. BAAI/bge-reranker-v2-m3)
    #     pass
    
    results = []
    for art_id, score in ranked:
        data = G.nodes[art_id]
        results.append({
            "id": art_id,
            "law_name": data.get("law_name"),
            "law_domain": data.get("law_domain"),
            "article_number": data.get("article_number"),
            "title": data.get("title"),
            "text_original": data.get("text_original"),
            "summary": data.get("summary", ""),
            "text_explanation": data.get("text_explanation", ""),
            "keywords": data.get("keywords", []),
            "graph_score": round(score, 4),
            "pagerank": round(PR.get(art_id, 0), 6),
        })
    return results

if __name__ == "__main__":
    q = "ما هي عقوبة السرقة في الجزائر؟"
    res = graph_retrieve(q)
    for r in res:
        print(f"[{r['id']}] {r['title']} (score: {r['graph_score']})")
