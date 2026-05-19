"""
step3_new_laws_triplets.py
==========================
Generates triplets for articles from newly added laws that the model
has never seen during training.

Steps:
  1. Walk dataset/raw/ and load all JSON article files → full corpus
  2. Identify articles whose law_name does NOT appear in the training dataset
  3. Embed new articles with current BGE-M3 (models/finetuned-bge-m3/)
  4. Find most similar existing article (cosine sim > 0.75) as hard negative
  5. Generate synthetic Arabic query from title + keywords
  6. Build triplets (positive = new article, negative = similar existing one)
  7. Target: 20-30 triplets covering new laws

Output:
  hard_negatives_new_laws.jsonl
  new_law_coverage_report.json

USAGE:
    python step3_new_laws_triplets.py
"""

import sys
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "step3_new_laws.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH    = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
RAW_DATA_DIR    = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_JSONL    = PROJECT_ROOT / "hard_negatives_new_laws.jsonl"
COVERAGE_JSON   = PROJECT_ROOT / "new_law_coverage_report.json"
MODEL_PATH      = PROJECT_ROOT / "models" / "finetuned-bge-m3"

SIM_THRESHOLD   = 0.75   # cosine similarity to accept a hard negative
MAX_TRIPLETS    = 30
SEED            = 42

random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — copied verbatim from mine_hard_negatives.py
# ══════════════════════════════════════════════════════════════════════════════

def _safe(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v is not None)
    return str(val)


def build_chunk_text(doc: dict) -> str:
    law_name    = _safe(doc.get("law_name"))
    art_num     = _safe(doc.get("article_number"))
    title       = _safe(doc.get("title"))
    keywords    = _safe(doc.get("keywords"))
    original    = _safe(doc.get("text_original"))
    explanation = _safe(doc.get("text_explanation"))
    summary     = _safe(doc.get("summary"))
    header = f"[{law_name} - المادة {art_num}]" if law_name and art_num else ""
    parts  = [p for p in [header, title, keywords, original, explanation, summary] if p]
    return " | ".join(parts).strip() or "unknown"


def law_code(doc: dict) -> str:
    name = str(doc.get("law_name", "")).strip()
    if "العقوبات" in name:                    return "penal"
    if "التجاري" in name:                     return "commercial"
    if "المدني" in name:                      return "civil"
    if "الاجراءات" in name:                   return "procedure"
    if "العمل" in name or "90-11" in name:    return "labour"
    if "18-05" in name:                       return "ecommerce"
    if "03-03" in name:                       return "competition"
    return name[:40]


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD RAW CORPUS — normalise the nested `text.original` field
# ══════════════════════════════════════════════════════════════════════════════

def normalise_article(raw: dict) -> dict:
    """
    Raw JSON files store text under raw['text']['original'].
    The pipeline uses flat key 'text_original'.
    This function normalises to the flat format used by build_chunk_text().
    """
    doc = dict(raw)
    # Flatten text field
    text_node = doc.get("text", {})
    if isinstance(text_node, dict):
        doc["text_original"] = text_node.get("original", "")
    else:
        doc["text_original"] = str(text_node)
    return doc


def load_raw_corpus() -> list[dict]:
    """Walk dataset/raw/ and load all JSON files."""
    articles = []
    for json_file in sorted(RAW_DATA_DIR.rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    articles.append(normalise_article(item))
            elif isinstance(data, dict):
                articles.append(normalise_article(data))
        except Exception as e:
            log.warning("Failed to load %s: %s", json_file.name, e)
    log.info("Loaded %d articles from raw corpus.", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
#  IDENTIFY COVERED LAW NAMES FROM TRAINING DATASET
# ══════════════════════════════════════════════════════════════════════════════

def get_covered_laws(dataset: list[dict]) -> set:
    """
    Extract all law names that appear in the training dataset's article references.
    Format of articles list: "القانون المدني - المادة 42"
    """
    covered = set()
    for item in dataset:
        for art_ref in item.get("articles", []):
            parts = art_ref.split(" - المادة ")
            if parts:
                law = parts[0].strip()
                if law:
                    covered.add(law)
    log.info("Found %d distinct law names in training dataset.", len(covered))
    return covered


def is_law_covered(law_name: str, covered_laws: set) -> bool:
    """Check if a law_name is represented in the training dataset."""
    ln = law_name.strip()
    for covered in covered_laws:
        if ln == covered or ln in covered or covered in ln:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC QUERY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# Colloquial triggers that make queries sound more natural (Algerian Arabic mix)
QUERY_TEMPLATES_FORMAL = [
    "ما هو حكم {concept} في القانون الجزائري؟",
    "ما هي شروط {concept}؟",
    "كيف ينظم القانون {concept}؟",
    "ما هي الإجراءات المتعلقة بـ{concept}؟",
    "ما هي حقوق الأطراف في {concept}؟",
    "ما هي أحكام {concept} وفق القانون الجزائري؟",
    "ما المقصود بـ{concept}؟",
    "هل يجوز {concept} قانونياً؟",
    "ما الفرق بين {concept} وما يشابهه؟",
    "ما هو دور القاضي في {concept}؟",
]

QUERY_TEMPLATES_COLLOQUIAL = [
    "كيفاش نفهم {concept}؟",
    "وين يمكن نلقى حكم {concept}؟",
    "علاش {concept} مهم في القانون؟",
    "واش {concept} مسموح به في الجزائر؟",
    "كيفاش يتم {concept} في الجزائر؟",
]


def generate_query(title: str, keywords: list, use_colloquial: bool = False) -> str:
    """
    Generate a natural Arabic question from article title and keywords.

    Rules:
    - Max 15 words
    - Uses title as primary concept, keywords as secondary
    - Occasionally uses colloquial triggers for variety
    """
    # Build concept: prefer short title, augment with first 2 keywords
    concept = title.strip()
    if len(concept.split()) > 6:
        # Use first 3-4 keywords as a more compact concept
        if keywords:
            kw_list = keywords[:3] if isinstance(keywords, list) else []
            concept = " و".join(kw_list[:2]) if kw_list else concept[:40]

    if use_colloquial and random.random() < 0.2:  # 20% colloquial
        template = random.choice(QUERY_TEMPLATES_COLLOQUIAL)
    else:
        template = random.choice(QUERY_TEMPLATES_FORMAL)

    query = template.format(concept=concept)

    # Enforce 15-word limit
    words = query.split()
    if len(words) > 15:
        query = " ".join(words[:15]) + "؟"

    return query


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING & SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and all rows of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("  Step 3 — New Law Article Triplet Generation")
    log.info("  Target: %d triplets  |  Sim threshold: %.2f", MAX_TRIPLETS, SIM_THRESHOLD)
    log.info("=" * 70)

    # ── Load training dataset to find covered laws ─────────────────────────────
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        training_dataset = json.load(f)
    covered_laws = get_covered_laws(training_dataset)

    # ── Load raw corpus ────────────────────────────────────────────────────────
    all_articles = load_raw_corpus()

    if not all_articles:
        log.error("No articles found in %s", RAW_DATA_DIR)
        return

    # ── Separate new vs existing articles ─────────────────────────────────────
    new_articles      = []
    existing_articles = []

    for art in all_articles:
        ln = str(art.get("law_name", "")).strip()
        if not ln:
            continue
        if is_law_covered(ln, covered_laws):
            existing_articles.append(art)
        else:
            new_articles.append(art)

    log.info("New articles (unseen laws)    : %d", len(new_articles))
    log.info("Existing articles (seen laws) : %d", len(existing_articles))

    if not new_articles:
        log.warning("No new articles found. All laws in corpus are covered by training dataset.")
        log.info("Tip: Add new JSON files to dataset/raw/ subdirectories.")
        # Save empty outputs
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            pass
        with open(COVERAGE_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    if not existing_articles:
        log.error("No existing articles found — cannot compute hard negatives.")
        return

    # ── Load BGE-M3 model ──────────────────────────────────────────────────────
    log.info("Loading BGE-M3 model from: %s", MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    model = SentenceTransformer(str(MODEL_PATH), device=device)

    # ── Build texts for embedding ──────────────────────────────────────────────
    log.info("Building chunk texts for %d new articles…", len(new_articles))
    new_texts      = [build_chunk_text(a) for a in new_articles]
    existing_texts = [build_chunk_text(a) for a in existing_articles]

    # ── Embed all articles ─────────────────────────────────────────────────────
    log.info("Embedding new articles (%d)…", len(new_articles))
    new_embeddings = model.encode(
        new_texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )

    log.info("Embedding existing articles (%d)…", len(existing_articles))
    existing_embeddings = model.encode(
        existing_texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )

    # ── Compute similarity matrix: new × existing ──────────────────────────────
    log.info("Computing cosine similarity matrix (%d × %d)…",
             len(new_articles), len(existing_articles))
    sim_matrix = cosine_similarity_matrix(new_embeddings, existing_embeddings)

    # ── Group new articles by law name ─────────────────────────────────────────
    law_groups: dict[str, list[int]] = {}
    for idx, art in enumerate(new_articles):
        ln = str(art.get("law_name", "")).strip()
        law_groups.setdefault(ln, []).append(idx)

    log.info("New laws discovered: %s", list(law_groups.keys()))

    # ── Generate triplets ──────────────────────────────────────────────────────
    triplets          = []
    coverage_report   = []
    used_new_indices  = set()

    for law_name, indices in law_groups.items():
        law_triplets = 0
        law_neg_sims = []

        # Shuffle for variety
        random.shuffle(indices)

        for new_idx in indices:
            if len(triplets) >= MAX_TRIPLETS:
                break
            if new_idx in used_new_indices:
                continue

            art       = new_articles[new_idx]
            art_text  = new_texts[new_idx]
            sim_row   = sim_matrix[new_idx]  # similarity to all existing articles

            # Find best negative: highest similarity, prefer different law code
            new_code = law_code(art)
            # Sort by (same_code_penalty, -similarity)
            sorted_existing = sorted(
                range(len(existing_articles)),
                key=lambda i: (law_code(existing_articles[i]) == new_code,
                               -float(sim_row[i]))
            )

            # Pick the best candidate above threshold
            neg_doc  = None
            neg_sim  = 0.0
            neg_text = None

            for ex_idx in sorted_existing:
                sim = float(sim_row[ex_idx])
                if sim >= SIM_THRESHOLD:
                    neg_doc  = existing_articles[ex_idx]
                    neg_sim  = sim
                    neg_text = existing_texts[ex_idx]
                    break

            if neg_doc is None:
                # Accept best available even below threshold (for coverage)
                best_ex_idx = int(np.argmax(sim_row))
                neg_sim     = float(sim_row[best_ex_idx])
                if neg_sim < 0.50:
                    log.debug("  Skipping %s Art.%s — best neg sim %.3f too low",
                              law_name[:30], art.get("article_number"), neg_sim)
                    continue
                neg_doc  = existing_articles[best_ex_idx]
                neg_text = existing_texts[best_ex_idx]
                log.info("  (Accepted below threshold) sim=%.3f for %s Art.%s",
                         neg_sim, law_name[:30], art.get("article_number"))

            # ── Generate synthetic query ───────────────────────────────────────
            title    = _safe(art.get("title"))
            keywords = art.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = [str(keywords)] if keywords else []

            use_colloquial = (len(triplets) % 5 == 4)  # every 5th query is colloquial
            query = generate_query(title, keywords, use_colloquial)

            if not query or not art_text or not neg_text:
                continue

            triplet = {
                "query": query,
                "pos"  : [art_text],
                "neg"  : [neg_text],
            }
            triplets.append(triplet)
            used_new_indices.add(new_idx)
            law_triplets += 1
            law_neg_sims.append(neg_sim)

            log.info(
                "  ✅ Triplet %d  law=%s  art=%s  neg_sim=%.3f",
                len(triplets), law_name[:30],
                art.get("article_number"), neg_sim,
            )
            log.info("     Q: %s", query)
            log.info("     Neg: %s - Art.%s",
                     neg_doc.get("law_name", "")[:30],
                     neg_doc.get("article_number", ""))

        avg_neg_sim = round(float(np.mean(law_neg_sims)), 4) if law_neg_sims else 0.0
        coverage_report.append({
            "law_name"          : law_name,
            "articles_total"    : len(indices),
            "triplets_generated": law_triplets,
            "avg_negative_similarity": avg_neg_sim,
        })

    # ── Save outputs ───────────────────────────────────────────────────────────
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for triplet in triplets:
            fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    with open(COVERAGE_JSON, "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  STEP 3 COMPLETE")
    log.info("=" * 70)
    log.info("  New laws found             : %d", len(law_groups))
    log.info("  New articles total         : %d", len(new_articles))
    log.info("  Triplets generated         : %d", len(triplets))
    log.info("  Triplets → %s", OUTPUT_JSONL.name)
    log.info("  Coverage → %s", COVERAGE_JSON.name)
    log.info("=" * 70)

    # ── Coverage table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 3 — NEW LAW COVERAGE REPORT")
    print("═" * 80)
    print(f"  {'Law Name':<45}  {'Articles':<9}  {'Triplets':<9}  {'Avg Neg Sim'}")
    print("─" * 80)
    for rec in coverage_report:
        print(
            f"  {rec['law_name']:<45}  {rec['articles_total']:<9}  "
            f"{rec['triplets_generated']:<9}  {rec['avg_negative_similarity']:.4f}"
        )
    print("═" * 80)
    print(f"  Total triplets: {len(triplets)}")
    print()

    # ── Show sample queries ────────────────────────────────────────────────────
    print("  Sample generated queries:")
    for t in triplets[:5]:
        print(f"    • {t['query']}")
    print()


if __name__ == "__main__":
    main()
