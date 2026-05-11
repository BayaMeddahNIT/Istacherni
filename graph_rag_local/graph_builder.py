"""
graph_builder.py
----------------
Builds an Algerian-law Knowledge Graph with local BGE-M3 embeddings.
This version computes embeddings for all article nodes and stores them in the graph.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

# Import local embeddings utility
from graph_rag_local.embeddings import embed_text

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
CACHE_DIR    = Path(__file__).parent / "cache"
# Use a separate cache file for the local embedding-enhanced graph
GRAPH_FILE   = CACHE_DIR / "law_graph_local.pkl"
CORPUS_FILE  = CACHE_DIR / "graph_corpus_local.pkl"
LAW_TYPE_CACHE_FILE = CACHE_DIR / "law_type_cache.json"


# ═══════════════════════════════════════════════════════════════════════
# ── Data loading (Purely from JSON metadata) ─────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _extract_text(article: dict) -> str:
    text_field = article.get("text")
    if isinstance(text_field, dict):
        original = text_field.get("original") or text_field.get("content") or ""
        if isinstance(original, dict):
            return " ".join(str(v) for v in original.values() if v)
        return str(original)
    if isinstance(text_field, str):
        return text_field
    return str(article.get("text_original", "") or "")


def _normalize(raw: dict) -> Optional[dict]:
    text = _extract_text(raw).strip()
    if not text:
        return None
    art_num_raw = raw.get("article_number", "")
    if isinstance(art_num_raw, list):
        art_num = "-".join(str(x) for x in art_num_raw)
    else:
        art_num = str(art_num_raw)

    relations = raw.get("relations", {}) or {}
    related_raw = relations.get("related_articles") or []
    related = [str(r) for r in related_raw if r is not None]

    # Combine metadata for a searchable text block (useful for embeddings)
    summary = raw.get("summary", "") or ""
    text_explanation = raw.get("text_explanation", "") or ""
    search_block = f"{raw.get('title', '')} {summary} {text} {' '.join(raw.get('keywords', []))}"

    return {
        "id":                       raw.get("id") or f"ART_{art_num}",
        "law_domain":               raw.get("law_domain", ""),
        "law_name":                 raw.get("law_name", ""),
        "article_number":           art_num,
        "title":                    raw.get("title", ""),
        "text_original":            text,
        "text_explanation":         text_explanation,
        "summary":                  summary,
        "search_block":             search_block.strip()[:2000], # Truncate for embedding efficiency
        "keywords":                 [k.strip() for k in raw.get("keywords", []) if k],
        "penalties_summary":        raw.get("penalties_summary", ""),
        "legal_conditions_summary": raw.get("legal_conditions_summary", ""),
        "related_articles":         related,
    }

def _remove_trailing_commas(text: str) -> str:
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text

def _load_articles(data_dir: Path = RAW_DATA_DIR) -> list[dict]:
    seen, articles = set(), []
    files = sorted(data_dir.rglob("*.json"))
    for path in files:
        if path.name.startswith("add_") or path.name.startswith("test"):
            continue
        content = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = path.read_text(encoding=enc)
                break
            except Exception:
                continue
        if content is None:
            continue

        raw_list = []
        for attempt, text in enumerate([content, _remove_trailing_commas(content)]):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    while len(data) == 1 and isinstance(data[0], list):
                        data = data[0]
                    raw_list = [item for item in data if isinstance(item, dict)]
                elif isinstance(data, dict):
                    raw_list = [data]
                break
            except json.JSONDecodeError:
                if attempt == 0: continue
                break

        for raw in raw_list:
            doc = _normalize(raw)
            if doc and doc["id"] not in seen:
                seen.add(doc["id"])
                articles.append(doc)
    return articles

def _classify_penalty(summary: str) -> list[str]:
    patterns = [
        (r"إعدام|الاعدام",            "الإعدام"),
        (r"سجن مؤبد|المؤبد",         "السجن المؤبد"),
        (r"سجن\s*\d",                 "السجن المؤقت"),
        (r"حبس",                      "الحبس"),
        (r"غرامة",                    "الغرامة المالية"),
        (r"مصادر",                    "المصادرة"),
        (r"حرمان من الحقوق",          "الحرمان من الحقوق"),
        (r"منع الاقامة",              "منع الإقامة"),
    ]
    if not summary: return ["غير محدد"]
    classes = [label for pattern, label in patterns if re.search(pattern, summary)]
    return classes if classes else ["غير محدد"]

_law_type_cache = {}
def _load_law_type_cache():
    global _law_type_cache
    if LAW_TYPE_CACHE_FILE.exists():
        with open(LAW_TYPE_CACHE_FILE, "r", encoding="utf-8") as f:
            _law_type_cache = json.load(f)

def _save_law_type_cache():
    with open(LAW_TYPE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_law_type_cache, f, ensure_ascii=False, indent=2)

def _classify_legal_type(art_id: str, title: str, summary: str, keywords: list[str]) -> str:
    global _law_type_cache
    if not _law_type_cache:
        _load_law_type_cache()
        
    if art_id in _law_type_cache:
        return _law_type_cache[art_id]

    text_to_check = f"{title} {summary} {' '.join(keywords)}".strip()
    
    # 1. Rule-based check
    procedural_keywords = ["إثبات", "دعوى", "محكمة", "رسمي", "توثيق", "إجراءات", "طعن", "قضاء", "دليل", "شهادة", "قاضي", "كاتب العدل", "محرر رسمي"]
    substantive_keywords = ["شروط", "أركان", "أهلية", "رضا", "سبب", "موضوع", "التزام", "عقد", "حق", "واجب", "بطلان", "فسخ", "إرادة"]
    
    p_score = sum(1 for w in procedural_keywords if w in text_to_check)
    s_score = sum(1 for w in substantive_keywords if w in text_to_check)
    
    # Rule engine priority
    if p_score > s_score + 1:
        res = "procedural"
    elif s_score > p_score + 1:
        res = "substantive"
    elif p_score == 0 and s_score == 0:
        res = "both"
    else:
        # 2. Fast CamelBERT Fallback for ambiguous articles instead of slow LLM
        try:
            from graph_rag_local.camelbert_classifier import get_camelbert_classifier
            classifier = get_camelbert_classifier()
            # CamelBERT returns the best domain. 
            # If the domain is Civil Law -> substantive, Penal/Procedural -> procedural, else both
            domain, score, _ = classifier.classify(text_to_check)
            if score > 0.4:
                if domain == "Civil Law":
                    res = "substantive"
                elif domain == "Code of Civil and Administrative Procedures":
                    res = "procedural"
                else:
                    res = "both"
            else:
                res = "both"
        except Exception as e:
            res = "both"
            
    _law_type_cache[art_id] = res
    _save_law_type_cache()
    return res


# ═══════════════════════════════════════════════════════════════════════
# ── Graph builder ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def build_graph(force: bool = False) -> tuple[nx.DiGraph, list[dict]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and GRAPH_FILE.exists() and CORPUS_FILE.exists():
        print("[GraphBuilder] Loading from cache…")
        with open(GRAPH_FILE,  "rb") as f: G = pickle.load(f)
        with open(CORPUS_FILE, "rb") as f: corpus = pickle.load(f)
        return G, corpus

    print("[GraphBuilder] Building new graph with BGE-M3 embeddings…")
    articles = _load_articles()
    G = nx.DiGraph()

    num_to_ids: dict[str, list] = {}
    law_to_ids: dict[str, list] = {}

    for art in articles:
        num_to_ids.setdefault(art["article_number"], []).append(art["id"])
        law_to_ids.setdefault(art["law_name"], []).append(art["id"])

    # ── Phase 1: Add nodes & compute embeddings ─────────────────────────
    print(f"[GraphBuilder] Phase 1 — Computing embeddings for {len(articles)} articles…")
    
    # Smart Ingestion: Reuse existing embeddings if search_block hasn't changed
    existing_embs = {}
    if GRAPH_FILE.exists():
        try:
            with open(GRAPH_FILE, "rb") as f:
                old_G = pickle.load(f)
                for n, d in old_G.nodes(data=True):
                    if d.get("node_type") == "article" and "embedding" in d:
                        existing_embs[d["search_block"]] = d["embedding"]
        except Exception:
            pass

    embeddings = [None] * len(articles)
    to_embed_indices = []
    texts_to_embed = []

    for i, art in enumerate(articles):
        block = art["search_block"]
        if block in existing_embs:
            embeddings[i] = existing_embs[block]
        else:
            to_embed_indices.append(i)
            texts_to_embed.append(block)

    if texts_to_embed:
        print(f"[GraphBuilder] Reusing {len(articles) - len(texts_to_embed)} embeddings, computing {len(texts_to_embed)} new ones.")
        new_embeddings = embed_text(texts_to_embed)
        for idx, emb in zip(to_embed_indices, new_embeddings):
            embeddings[idx] = emb
    else:
        print(f"[GraphBuilder] Reusing all {len(articles)} existing embeddings.")

    for i, art in enumerate(articles):
        # Article node — store ALL metadata fields for downstream generation
        G.add_node(
            art["id"],
            node_type        = "article",
            law_name         = art["law_name"],
            law_domain       = art["law_domain"],
            article_number   = art["article_number"],
            title            = art["title"],
            text_original    = art["text_original"],
            text_explanation = art["text_explanation"],
            summary          = art["summary"],
            search_block     = art["search_block"],
            embedding        = embeddings[i],  # Store the BGE-M3 embedding!
            penalties        = _classify_penalty(art["penalties_summary"]),
            keywords         = art["keywords"],
            law_type         = _classify_legal_type(art["id"], art.get("title", ""), art.get("summary", ""), art.get("keywords", [])),
        )

        for kw in art["keywords"]:
            cid = f"CONCEPT:{kw}"
            if not G.has_node(cid):
                G.add_node(cid, node_type="concept", term=kw)

    # ── Phase 2: Add edges ─────────────────────────────────────────────
    print("[GraphBuilder] Phase 2 — Adding edges…")
    for art in articles:
        art_id = art["id"]
        for kw in art["keywords"]:
            G.add_edge(art_id, f"CONCEPT:{kw}", rel="HAS_KEYWORD")
        
        did = f"DOMAIN:{art['law_domain']}"
        if not G.has_node(did): G.add_node(did, node_type="domain", name=art["law_domain"])
        G.add_edge(art_id, did, rel="IN_DOMAIN")

        for p_class in _classify_penalty(art["penalties_summary"]):
            pid = f"PENALTY:{p_class}"
            if not G.has_node(pid): G.add_node(pid, node_type="penalty", penalty_class=p_class)
            G.add_edge(art_id, pid, rel="HAS_PENALTY")

        for rel_num in art["related_articles"]:
            for cand_id in num_to_ids.get(rel_num, []):
                if cand_id != art_id:
                    G.add_edge(art_id, cand_id, rel="RELATED_TO")

    # SAME_LAW edges
    for law_name, ids in law_to_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                G.add_edge(ids[i], ids[j], rel="SAME_LAW", weight=0.3)
                G.add_edge(ids[j], ids[i], rel="SAME_LAW", weight=0.3)

    # Persist
    with open(GRAPH_FILE,  "wb") as f: pickle.dump(G, f)
    with open(CORPUS_FILE, "wb") as f: pickle.dump(articles, f)
    print(f"[GraphBuilder] * Graph built and cached with BGE-M3 embeddings.")
    return G, articles

if __name__ == "__main__":
    build_graph(force=True)
