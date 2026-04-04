"""
graph_builder.py
----------------
Builds an Algerian-law Knowledge Graph using NetworkX — NO API CALLS needed.
Everything is derived purely from the structured JSON metadata already in your dataset.

Graph anatomy
─────────────
NODE TYPES
  • article   →  one node per law article (id = article's JSON id)
  • concept   →  one node per keyword/legal-term (id = "CONCEPT:<term>")
  • domain    →  one node per law domain       (id = "DOMAIN:<domain>")
  • penalty   →  one node per penalty class     (id = "PENALTY:<class>")

EDGE TYPES
  • HAS_KEYWORD    article  → concept      (from "keywords" field)
  • RELATED_TO     article  → article      (from "relations.related_articles")
  • IN_DOMAIN      article  → domain       (from "law_domain")
  • HAS_PENALTY    article  → penalty      (derived from "penalties_summary" text)
  • SAME_LAW       article  → article      (articles sharing the same law_name)

Usage:
  python -m graph_rag.graph_builder          # build & save the graph
  from graph_rag.graph_builder import load_graph
  G = load_graph()
"""

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
CACHE_DIR    = Path(__file__).parent / "cache"
GRAPH_FILE   = CACHE_DIR / "law_graph.pkl"
CORPUS_FILE  = CACHE_DIR / "graph_corpus.pkl"


# ═══════════════════════════════════════════════════════════════════════
# ── Data loading (identical to BM25/Agentic loaders but standalone) ──
# ═══════════════════════════════════════════════════════════════════════

def _extract_text(article: dict) -> str:
    if isinstance(article.get("text"), dict):
        return article["text"].get("original", "")
    if isinstance(article.get("text"), str):
        return article["text"]
    return article.get("text_original", "")


def _normalize(raw: dict) -> Optional[dict]:
    text = _extract_text(raw).strip()
    if not text:
        return None
    art_num = str(raw.get("article_number", ""))

    # Collect related article numbers (may be stored as ints or strings)
    relations = raw.get("relations", {}) or {}
    related_raw = relations.get("related_articles") or []
    related = [str(r) for r in related_raw if r is not None]

    return {
        "id":                       raw.get("id") or f"ART_{art_num}",
        "law_domain":               raw.get("law_domain", ""),
        "law_name":                 raw.get("law_name", ""),
        "article_number":           art_num,
        "title":                    raw.get("title", ""),
        "text_original":            text,
        "summary":                  raw.get("summary", ""),
        "keywords":                 [k.strip() for k in raw.get("keywords", []) if k],
        "penalties_summary":        raw.get("penalties_summary", ""),
        "legal_conditions_summary": raw.get("legal_conditions_summary", ""),
        "related_articles":         related,
    }


def _load_articles(data_dir: Path = RAW_DATA_DIR) -> list[dict]:
    print(f"[Debug] Searching for files in: {data_dir.absolute()}") # Add this line
    seen, articles = set(), []
    files = sorted(data_dir.rglob("*.json")) + sorted(data_dir.rglob("*.jsonl"))
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
        if path.suffix == ".jsonl":
            raw_list = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_list.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            # If no valid lines parsed, try the whole content as JSON array
            if not raw_list:
                try:
                    data = json.loads(content)
                    raw_list = data if isinstance(data, list) else [data]
                except json.JSONDecodeError:
                    pass
        else:
            try:
                data = json.loads(content)
                raw_list = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                continue
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            doc = _normalize(raw)
            if doc and doc["id"] not in seen:
                seen.add(doc["id"])
                articles.append(doc)
    return articles


# ═══════════════════════════════════════════════════════════════════════
# ── Penalty classifier ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

_PENALTY_PATTERNS = [
    (r"إعدام|الاعدام",            "الإعدام"),
    (r"سجن مؤبد|المؤبد",         "السجن المؤبد"),
    (r"سجن\s*\d",                 "السجن المؤقت"),
    (r"حبس",                      "الحبس"),
    (r"غرامة",                    "الغرامة المالية"),
    (r"مصادر",                    "المصادرة"),
    (r"حرمان من الحقوق",          "الحرمان من الحقوق"),
    (r"منع الاقامة",              "منع الإقامة"),
]

def _classify_penalty(summary: str) -> list[str]:
    if not summary:
        return ["غير محدد"]
    classes = []
    for pattern, label in _PENALTY_PATTERNS:
        if re.search(pattern, summary):
            classes.append(label)
    return classes if classes else ["غير محدد"]


# ═══════════════════════════════════════════════════════════════════════
# ── Graph builder ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def build_graph(force: bool = False) -> tuple[nx.DiGraph, list[dict]]:
    """
    Build the Knowledge Graph and persist it to disk.

    Args:
        force: If True, rebuild even if cache exists.

    Returns:
        (G, corpus)   G = networkx.DiGraph; corpus = list of article dicts.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and GRAPH_FILE.exists() and CORPUS_FILE.exists():
        print("[GraphBuilder] Loading graph from cache…")
        return _load_graph()

    print("[GraphBuilder] Loading articles from dataset…")
    articles = _load_articles()
    print(f"[GraphBuilder] {len(articles)} articles loaded.")

    G = nx.DiGraph()

    # ── Build an ID→article map (needed for RELATED_TO edges) ─────────
    # Also build a law_name → [id, …] map for SAME_LAW edges
    id_map:        dict[str, dict] = {}
    num_to_ids:    dict[str, list] = {}   # article_number → [ids sharing that number]
    law_to_ids:    dict[str, list] = {}

    for art in articles:
        id_map[art["id"]] = art
        num = art["article_number"]
        num_to_ids.setdefault(num, []).append(art["id"])
        law_to_ids.setdefault(art["law_name"], []).append(art["id"])

    # ── Phase 1: Add all nodes ─────────────────────────────────────────
    print("[GraphBuilder] Phase 1 — Adding nodes…")
    for art in articles:
        # Article node
        G.add_node(
            art["id"],
            node_type    = "article",
            law_name     = art["law_name"],
            law_domain   = art["law_domain"],
            article_number = art["article_number"],
            title        = art["title"],
            text_original = art["text_original"],
            summary      = art["summary"],
            penalties_summary = art["penalties_summary"],
            legal_conditions_summary = art["legal_conditions_summary"],
            keywords     = art["keywords"],
        )

        # Concept nodes (keywords)
        for kw in art["keywords"]:
            cid = f"CONCEPT:{kw}"
            if not G.has_node(cid):
                G.add_node(cid, node_type="concept", term=kw)

        # Domain node
        did = f"DOMAIN:{art['law_domain']}"
        if not G.has_node(did):
            G.add_node(did, node_type="domain", name=art["law_domain"])

        # Penalty nodes
        for p_class in _classify_penalty(art["penalties_summary"]):
            pid = f"PENALTY:{p_class}"
            if not G.has_node(pid):
                G.add_node(pid, node_type="penalty", penalty_class=p_class)

    # ── Phase 2: Add all edges ─────────────────────────────────────────
    print("[GraphBuilder] Phase 2 — Adding edges…")
    edges = 0
    for art in articles:
        art_id = art["id"]

        # HAS_KEYWORD edges
        for kw in art["keywords"]:
            G.add_edge(art_id, f"CONCEPT:{kw}", rel="HAS_KEYWORD")
            edges += 1

        # IN_DOMAIN edges
        G.add_edge(art_id, f"DOMAIN:{art['law_domain']}", rel="IN_DOMAIN")
        edges += 1

        # HAS_PENALTY edges
        for p_class in _classify_penalty(art["penalties_summary"]):
            G.add_edge(art_id, f"PENALTY:{p_class}", rel="HAS_PENALTY")
            edges += 1

        # RELATED_TO edges — match by article_number across same law
        for rel_num in art["related_articles"]:
            for candidate_id in num_to_ids.get(rel_num, []):
                if candidate_id != art_id:
                    G.add_edge(art_id, candidate_id, rel="RELATED_TO")
                    edges += 1

    # ── SAME_LAW edges (articles in the same law are loosely connected) ──
    for law_name, ids in law_to_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                G.add_edge(ids[i], ids[j], rel="SAME_LAW", weight=0.3)
                G.add_edge(ids[j], ids[i], rel="SAME_LAW", weight=0.3)
                edges += 2

    print(f"[GraphBuilder] Graph complete:")
    print(f"  Nodes : {G.number_of_nodes()} ({len(articles)} articles + concept/domain/penalty nodes)")
    print(f"  Edges : {G.number_of_edges()}")

    # Persist
    with open(GRAPH_FILE,  "wb") as f:
        pickle.dump(G, f)
    with open(CORPUS_FILE, "wb") as f:
        pickle.dump(articles, f)
    print(f"[GraphBuilder] ✓ Saved → {GRAPH_FILE}\n")

    return G, articles


def _load_graph() -> tuple[nx.DiGraph, list[dict]]:
    with open(GRAPH_FILE,  "rb") as f:
        G = pickle.load(f)
    with open(CORPUS_FILE, "rb") as f:
        corpus = pickle.load(f)
    print(f"[GraphBuilder] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G, corpus


# expose as convenience
def load_graph() -> tuple[nx.DiGraph, list[dict]]:
    return build_graph(force=False)


# ── Stats ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G, corpus = build_graph(force=True)
    print("\n── Node type counts ──")
    from collections import Counter
    types = Counter(d["node_type"] for _, d in G.nodes(data=True))
    for t, cnt in types.most_common():
        print(f"  {t:10s}: {cnt}")

    print("\n── Top 10 most connected article nodes ──")
    article_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "article"]
    by_degree = sorted(article_nodes, key=lambda n: G.degree(n), reverse=True)[:10]
    for n in by_degree:
        d = G.nodes[n]
        print(f"  degree={G.degree(n):4d}  {n}  [{d.get('title','')[:50]}]")
