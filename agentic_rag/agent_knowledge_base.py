"""
agent_knowledge_base.py
-----------------------
Standalone knowledge base for the Agentic RAG.
Loads all JSON/JSONL articles from dataset/raw/** into memory and exposes
three retrieval primitives that the agent can call as tools:

  1. search_articles(query, top_k)       → BM25 keyword search
  2. filter_by_domain(domain, query, top_k) → restrict to one law domain, then BM25
  3. get_article_by_id(article_id)       → direct article look-up by its ID

No external dependencies beyond rank_bm25 and numpy.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DATA_DIR  = PROJECT_ROOT / "dataset" / "raw"
CACHE_DIR     = Path(__file__).parent / "cache"
INDEX_FILE    = CACHE_DIR / "agent_bm25.pkl"
CORPUS_FILE   = CACHE_DIR / "agent_corpus.pkl"


# ═══════════════════════════════════════════════════════════════════════
# ── Arabic tokeniser ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Strip diacritics, normalise Arabic letter variants, split into tokens."""
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)   # remove tashkeel / tatweel
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ة",       "ه", text)
    text = re.sub(r"ى",       "ي", text)
    tokens = re.split(r"[^\w\u0600-\u06FF]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ═══════════════════════════════════════════════════════════════════════
# ── Data loader ─────────────────────────────────────────────────────
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
    return {
        "id":                       raw.get("id") or f"ART_{art_num}",
        "law_domain":               raw.get("law_domain", ""),
        "law_name":                 raw.get("law_name", ""),
        "article_number":           art_num,
        "title":                    raw.get("title", ""),
        "text_original":            text,
        "summary":                  raw.get("summary", ""),
        "keywords":                 raw.get("keywords", []),
        "penalties_summary":        raw.get("penalties_summary", ""),
        "legal_conditions_summary": raw.get("legal_conditions_summary", ""),
    }


def _load_raw_articles(data_dir: Path) -> list[dict]:
    """Load and normalise every *.json / *.jsonl in data_dir recursively."""
    seen, articles = set(), []
    files = sorted(data_dir.rglob("*.json")) + sorted(data_dir.rglob("*.jsonl"))
    for path in files:
        if path.name.startswith("add_") or path.name.startswith("test"):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue

        if path.suffix == ".jsonl":
            raw_list = [json.loads(l) for l in content.splitlines() if l.strip()]
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


def _doc_text(a: dict) -> str:
    kw = " ".join(a.get("keywords", []))
    return " ".join([
        a.get("title", ""), a.get("text_original", ""),
        a.get("summary", ""), a.get("legal_conditions_summary", ""),
        a.get("penalties_summary", ""), kw, kw,   # keywords ×2 boost
    ])


# ═══════════════════════════════════════════════════════════════════════
# ── Knowledge Base (singleton) ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

class _KnowledgeBase:
    """Singleton that holds the corpus and BM25 index in memory."""

    def __init__(self):
        self._corpus: list[dict] = []
        self._bm25:   Optional[BM25Okapi] = None
        self._by_id:  dict[str, dict] = {}
        self._by_domain: dict[str, list[dict]] = {}

    def _ensure_loaded(self):
        if self._bm25 is not None:
            return

        # Try loading from disk cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if INDEX_FILE.exists() and CORPUS_FILE.exists():
            with open(INDEX_FILE, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(CORPUS_FILE, "rb") as f:
                self._corpus = pickle.load(f)
            print(f"[AgentKB] Loaded {len(self._corpus)} articles from cache.")
        else:
            print("[AgentKB] Building index from dataset…")
            articles = _load_raw_articles(RAW_DATA_DIR)
            tokenized = [_tokenize(_doc_text(a)) for a in articles]
            valid = [(tok, art) for tok, art in zip(tokenized, articles) if tok]
            tok_v, art_v = zip(*valid) if valid else ([], [])
            self._bm25   = BM25Okapi(list(tok_v))
            self._corpus = list(art_v)
            with open(INDEX_FILE, "wb") as f:
                pickle.dump(self._bm25, f)
            with open(CORPUS_FILE, "wb") as f:
                pickle.dump(self._corpus, f)
            print(f"[AgentKB] Index built: {len(self._corpus)} docs, {len(self._bm25.idf)} terms.")

        # Build fast look-up caches
        for art in self._corpus:
            self._by_id[art["id"]] = art
            d = art.get("law_domain", "Unknown")
            self._by_domain.setdefault(d, []).append(art)

    # ── Tool 1: full-corpus BM25 search ───────────────────────────────
    def search_articles(self, query: str, top_k: int = 5) -> list[dict]:
        self._ensure_loaded()
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            s = float(scores[idx])
            if s <= 0:
                continue
            art = dict(self._corpus[idx])
            art["bm25_score"] = round(s, 4)
            results.append(art)
        return results

    # ── Tool 2: domain-restricted BM25 search ─────────────────────────
    def filter_by_domain(self, domain: str, query: str, top_k: int = 5) -> list[dict]:
        self._ensure_loaded()
        # Fuzzy domain match (case-insensitive, partial)
        domain_lower = domain.lower()
        pool = [
            a for key, arts in self._by_domain.items()
            for a in arts
            if domain_lower in key.lower()
        ]
        if not pool:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return pool[:top_k]
        # Build a local BM25 over the domain subset
        tokenized_pool = [_tokenize(_doc_text(a)) for a in pool]
        local_bm25 = BM25Okapi([t if t else [""] for t in tokenized_pool])
        scores = local_bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            s = float(scores[idx])
            art = dict(pool[idx])
            art["bm25_score"] = round(s, 4)
            results.append(art)
        return results

    # ── Tool 3: direct article look-up ────────────────────────────────
    def get_article_by_id(self, article_id: str) -> Optional[dict]:
        self._ensure_loaded()
        return self._by_id.get(article_id)

    # ── Utility ─────────────────────────────────────────────────────
    def available_domains(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._by_domain.keys())


# Singleton
KB = _KnowledgeBase()
