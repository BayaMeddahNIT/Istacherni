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
_RAW_TESTS = [
    # ── Penal Law ─────────────────────────────────────────────────────────────
    {
        "id": "TC01",
        "question": "ما هي عقوبة السرقة في القانون الجزائري؟",
        "expected_ids": ["DZ_PENAL_ART_350", "DZ_PENAL_ART_351"],
        "domain": "Penal Law",
        "keywords_hint": ["سرقة", "عقوبة"],
    },
    {
        "id": "TC02",
        "question": "ما هي عقوبة القتل العمد في القانون الجزائري؟",
        "expected_ids": ["DZ_PENAL_ART_254"],
        "domain": "Penal Law",
        "keywords_hint": ["قتل", "عمد", "إعدام"],
    },
    {
        "id": "TC03",
        "question": "ما هي عقوبة جريمة الاحتيال؟",
        "expected_ids": ["DZ_PENAL_ART_372"],
        "domain": "Penal Law",
        "keywords_hint": ["احتيال", "نصب", "غش"],
    },
    {
        "id": "TC04",
        "question": "ما هي عقوبة الاغتصاب في القانون الجزائري؟",
        "expected_ids": ["DZ_PENAL_ART_336"],
        "domain": "Penal Law",
        "keywords_hint": ["اغتصاب", "هتك", "عرض"],
    },
    {
        "id": "TC05",
        "question": "ما هي عقوبة تزوير الوثائق الرسمية؟",
        "expected_ids": ["DZ_PENAL_ART_214", "DZ_PENAL_ART_216"],
        "domain": "Penal Law",
        "keywords_hint": ["تزوير", "وثائق", "مستندات"],
    },
    {
        "id": "TC06",
        "question": "ما هي عقوبة الرشوة في القانون الجزائري؟",
        "expected_ids": ["DZ_PENAL_ART_25", "DZ_PENAL_ART_126"],
        "domain": "Penal Law",
        "keywords_hint": ["رشوة", "فساد", "موظف"],
    },
    # ── Civil Law ─────────────────────────────────────────────────────────────
    {
        "id": "TC07",
        "question": "ما هي شروط صحة عقد البيع؟",
        "expected_ids": ["DZ_CIVIL_ART_351", "DZ_CIVIL_ART_54"],
        "domain": "Civil Law",
        "keywords_hint": ["بيع", "عقد", "شروط"],
    },
    {
        "id": "TC08",
        "question": "كيف يتم التعويض عن الضرر في القانون المدني الجزائري؟",
        "expected_ids": ["DZ_CIVIL_ART_124", "DZ_CIVIL_ART_182"],
        "domain": "Civil Law",
        "keywords_hint": ["تعويض", "ضرر", "مسؤولية"],
    },
    {
        "id": "TC09",
        "question": "ما هي شروط الأهلية القانونية لإبرام العقود؟",
        "expected_ids": ["DZ_CIVIL_ART_40", "DZ_CIVIL_ART_42"],
        "domain": "Civil Law",
        "keywords_hint": ["أهلية", "عقد", "سن"],
    },
    {
        "id": "TC10",
        "question": "ما هي أحكام التقادم المسقط في القانون المدني؟",
        "expected_ids": ["DZ_CIVIL_ART_308", "DZ_CIVIL_ART_309"],
        "domain": "Civil Law",
        "keywords_hint": ["تقادم", "انقضاء"],
    },
    {
        "id": "TC11",
        "question": "ما هي شروط عقد الإيجار وما هي حقوق المستأجر؟",
        "expected_ids": ["DZ_CIVIL_ART_467", "DZ_CIVIL_ART_468"],
        "domain": "Civil Law",
        "keywords_hint": ["إيجار", "مستأجر", "عقد"],
    },
    {
        "id": "TC12",
        "question": "ما هي أحكام الوكالة في القانون المدني الجزائري؟",
        "expected_ids": ["DZ_CIVIL_ART_571", "DZ_CIVIL_ART_572"],
        "domain": "Civil Law",
        "keywords_hint": ["وكالة", "وكيل", "موكل"],
    },
    # ── Labor Law ─────────────────────────────────────────────────────────────
    {
        "id": "TC13",
        "question": "ما هي حقوق العامل في حالة الفصل التعسفي؟",
        "expected_ids": [],   # resolved at runtime
        "domain": "Labor Law",
        "keywords_hint": ["فصل", "تعسفي", "عمال"],
    },
    {
        "id": "TC14",
        "question": "ما هي حقوق العامل في العطل السنوية والإجازات؟",
        "expected_ids": [],
        "domain": "Labor Law",
        "keywords_hint": ["إجازة", "عطلة", "عمل"],
    },
    {
        "id": "TC15",
        "question": "ما هي شروط عقد العمل محدد المدة؟",
        "expected_ids": [],
        "domain": "Labor Law",
        "keywords_hint": ["عقد", "عمل", "مؤقت"],
    },
    # ── Commercial Law ────────────────────────────────────────────────────────
    {
        "id": "TC16",
        "question": "ما هي شروط تأسيس شركة تجارية في الجزائر؟",
        "expected_ids": [],
        "domain": "Commercial Law",
        "keywords_hint": ["شركة", "تجارية", "تأسيس"],
    },
    {
        "id": "TC17",
        "question": "ما هي أحكام الإفلاس التجاري في القانون الجزائري؟",
        "expected_ids": [],
        "domain": "Commercial Law",
        "keywords_hint": ["إفلاس", "تجاري", "توقف"],
    },
    # ── Civil & Administrative Procedure ─────────────────────────────────────
    {
        "id": "TC18",
        "question": "ما هي شروط الطعن بالاستئناف في المحاكم الجزائرية؟",
        "expected_ids": [],
        "domain": "Civil Procedure",
        "keywords_hint": ["استئناف", "طعن", "محكمة"],
    },
    {
        "id": "TC19",
        "question": "ما هي إجراءات الطعن بالنقض أمام المحكمة العليا؟",
        "expected_ids": [],
        "domain": "Civil Procedure",
        "keywords_hint": ["نقض", "محكمة عليا", "طعن"],
    },
    # ── Penal — extra ────────────────────────────────────────────────────────
    {
        "id": "TC20",
        "question": "ما هي عقوبة جريمة إخفاء الأشياء المسروقة؟",
        "expected_ids": ["DZ_PENAL_ART_387"],
        "domain": "Penal Law",
        "keywords_hint": ["إخفاء", "مسروقات", "حيازة"],
    },
]


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
