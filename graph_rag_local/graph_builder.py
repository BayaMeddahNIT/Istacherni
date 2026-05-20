import json
import pickle
import re
import sys
import io
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

# Force UTF-8 for terminal output (Fixes Arabic encoding crashes on Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import local embeddings utility
from graph_rag_local.embeddings import embed_text

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
CACHE_DIR    = Path(__file__).parent / "cache"
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

    title            = raw.get("title", "") or ""
    text_explanation = raw.get("text_explanation", "") or ""
    summary          = raw.get("summary", "") or ""
    search_block     = f"{title}. {text}".strip() if title else text

    return {
        "id":                       raw.get("id") or f"ART_{art_num}",
        "law_domain":               raw.get("law_domain", ""),
        "law_name":                 raw.get("law_name", ""),
        "article_number":           art_num,
        "title":                    title,
        "text_original":            text,
        "text_explanation":         text_explanation,
        "summary":                  summary,
        "search_block":             search_block[:2000],
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAW_TYPE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_law_type_cache, f, ensure_ascii=False, indent=4)

def get_law_type(art_id: str, text_to_check: str) -> str:
    if art_id in _law_type_cache:
        return _law_type_cache[art_id]
    
    # Simple rule-based scoring
    p_keywords = ["إجراءات", "ميعاد", "تبليغ", "اختصاص", "خصومة", "طعن", "نقض", "استئناف", "قضائية", "محضر"]
    s_keywords = ["حق", "التزام", "عقد", "بيع", "ملكية", "مسؤولية", "ضرر", "تعويض", "عقوبة", "جريمة", "حبس", "سجن"]
    
    p_score = sum(1 for k in p_keywords if k in text_to_check)
    s_score = sum(1 for k in s_keywords if k in text_to_check)
    
    if p_score > s_score + 1:
        res = "procedural"
    elif s_score > p_score + 1:
        res = "substantive"
    elif p_score == 0 and s_score == 0:
        res = "both"
    else:
        try:
            from graph_rag_local.camelbert_classifier import get_camelbert_classifier
            classifier = get_camelbert_classifier()
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
        except Exception:
            res = "both"
            
    _law_type_cache[art_id] = res
    _save_law_type_cache()
    return res

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

    print(f"[GraphBuilder] Phase 1 — Computing embeddings for {len(articles)} articles…")
    
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
        new_embs = embed_text(texts_to_embed)
        for idx, emb in zip(to_embed_indices, new_embs):
            embeddings[idx] = emb
    else:
        print(f"[GraphBuilder] Reusing all {len(articles)} embeddings.")

    for art, emb in zip(articles, embeddings):
        art_id = art["id"]
        law_type = get_law_type(art_id, art["search_block"])
        G.add_node(art_id, node_type="article", embedding=emb, law_type=law_type, **art)

    print("[GraphBuilder] Phase 2 — Adding edges…")
    for art_id in G.nodes():
        node_data = G.nodes[art_id]
        
        # 1. Statutory relations
        for rel_id in node_data.get("related_articles", []):
            targets = num_to_ids.get(rel_id, [])
            for target in targets:
                if target != art_id:
                    G.add_edge(art_id, target, relationship="related")

        # 2. Domain consistency (edges between articles in same law)
        # law_name = node_data.get("law_name")
        # if law_name:
        #     siblings = law_to_ids.get(law_name, [])
        #     for sib in siblings:
        #         if sib != art_id:
        #             G.add_edge(art_id, sib, relationship="same_law")

    with open(GRAPH_FILE, "wb") as f: pickle.dump(G, f)
    with open(CORPUS_FILE, "wb") as f: pickle.dump(articles, f)
    print("[GraphBuilder] * Graph built and cached with BGE-M3 embeddings.")
    return G, articles

if __name__ == "__main__":
    build_graph(force=True)
