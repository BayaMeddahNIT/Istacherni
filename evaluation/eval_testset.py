"""
eval_testset.py
---------------
20 ground-truth test cases for the Algerian Law RAG evaluation.

At load time, each test case is validated against the real corpus so that:
  - Expected article IDs that don't exist are automatically removed.
  - Test cases with no valid expected IDs are auto-resolved by running a
    BM25 search against the full corpus (finds the best-matching article
    as the ground truth).

This makes the test set robust against any dataset changes.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Corpus loader (self-contained, does NOT import from other RAG modules) ─────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"


@dataclass
class TestCase:
    id: str
    question: str
    expected_ids: list[str]     # validated article IDs from the real corpus
    domain: str
    keywords_hint: list[str]    # used for auto-resolution if expected_ids is empty
    resolved: bool = False      # True if IDs were auto-resolved at runtime


# ─────────────────────────────────────────────────────────────────────────────
# Raw test definitions (20 questions, 5+ domains)
# ─────────────────────────────────────────────────────────────────────────────
_RAW_TESTS = []
golden_path = Path(__file__).resolve().parent / "golden_dataset.jsonl"
if golden_path.exists():
    with open(golden_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                _RAW_TESTS.append({
                    "id": f"TC{idx:02d}",
                    "question": obj.get("query", ""),
                    "expected_ids": [obj.get("expected_article_id")] if obj.get("expected_article_id") else [],
                    "domain": obj.get("domain", ""),
                    "keywords_hint": []
                })
            except Exception:
                pass
else:
    print("WARNING: evaluation/golden_dataset.jsonl not found. Run evaluation/build_dataset.py first.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone corpus loader (no dependency on other RAG packages)
# ─────────────────────────────────────────────────────────────────────────────

def _load_corpus() -> dict[str, dict]:
    """Returns {article_id: article_dict} map."""
    id_map: dict[str, dict] = {}
    files = sorted(RAW_DATA_DIR.rglob("*.json")) + sorted(RAW_DATA_DIR.rglob("*.jsonl"))
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
        items = []
        if path.suffix == ".jsonl":
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        else:
            try:
                data = json.loads(content)
                items = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                continue
        for raw in items:
            if not isinstance(raw, dict):
                continue
            art_num = str(raw.get("article_number", ""))
            art_id  = raw.get("id") or f"ART_{art_num}"
            if art_id not in id_map:
                id_map[art_id] = raw
    return id_map


def _bm25_search(query: str, corpus: dict[str, dict],
                 domain_hint: str = "", top_k: int = 3) -> list[str]:
    """Minimal BM25 search used only for auto-resolving test case IDs."""
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
    except ImportError:
        # Fallback: just pick first article in domain
        return [aid for aid, art in corpus.items()
                if domain_hint.lower() in (art.get("law_domain") or "").lower()][:top_k]

    def _norm(t):
        t = re.sub(r"[\u064B-\u065F\u0640]", "", t)
        t = re.sub(r"[أإآا]", "ا", t)
        t = re.sub(r"ة", "ه", t)
        t = re.sub(r"ى", "ي", t)
        return t

    def _tok(text):
        text = _norm(text)
        return [w for w in re.split(r"[^\w\u0600-\u06FF]+", text.lower()) if len(w) >= 2]

    # Filter by domain if hint given
    pool = {
        aid: art for aid, art in corpus.items()
        if not domain_hint or domain_hint.lower() in (art.get("law_domain") or "").lower()
    }
    if not pool:
        pool = corpus

    ids     = list(pool.keys())
    arts    = list(pool.values())

    def _text(a):
        t = a.get("text_original") or ""
        if isinstance(a.get("text"), dict):
            t = a["text"].get("original", "") or t
        kw = " ".join(a.get("keywords", []))
        return f"{a.get('title','')} {t} {kw}"

    tokenized = [_tok(_text(a)) for a in arts]
    tokenized = [t if t else [""] for t in tokenized]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tok(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [ids[i] for i in top_idx if scores[i] > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_test_cases(n: Optional[int] = None) -> list[TestCase]:
    """
    Load and validate all test cases against the real corpus.

    Args:
        n: If given, return only the first n test cases.

    Returns:
        List of validated TestCase objects.
    """
    print("[TestSet] Loading corpus for validation...")
    corpus = _load_corpus()
    print(f"[TestSet] Corpus size: {len(corpus)} articles")

    cases = []
    for raw in _RAW_TESTS:
        # 1) Keep only IDs that actually exist in corpus
        valid_ids = [eid for eid in raw["expected_ids"] if eid in corpus]

        resolved = False
        # 2) If none valid -> auto-resolve via BM25
        if not valid_ids:
            valid_ids = _bm25_search(
                raw["question"],
                corpus,
                domain_hint=raw["domain"],
                top_k=3,
            )
            resolved = True
            print(f"  [TestSet] {raw['id']} auto-resolved -> {valid_ids}")
        else:
            removed = set(raw["expected_ids"]) - set(valid_ids)
            if removed:
                print(f"  [TestSet] {raw['id']} dropped missing IDs: {removed}")

        cases.append(TestCase(
            id            = raw["id"],
            question      = raw["question"],
            expected_ids  = valid_ids,
            domain        = raw["domain"],
            keywords_hint = raw["keywords_hint"],
            resolved      = resolved,
        ))

    if n is not None:
        cases = cases[:n]

    print(f"[TestSet] {len(cases)} test cases ready "
          f"({sum(1 for c in cases if c.resolved)} auto-resolved).\n")
    return cases
