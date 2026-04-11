# -*- coding: utf-8 -*-
"""
test_joradp_pipeline.py — Verification Test
============================================
Runs the full JORADP pipeline on the first 5 issues of 2025
and produces a sample JSONL artifact for inspection.

Usage:
  python test_joradp_pipeline.py
"""

import io as _io
import sys
import json
import logging
from pathlib import Path

# UTF-8 safe stdout wrapper for Windows CP1252 consoles
_safe_stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Patch sys.path so we can import from scraper.py ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from scraper import (
    JORADPScraper,
    PDFExtractor,
    GeminiParser,
    OUTPUT_DIR,
    PDF_CACHE_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(_safe_stdout)],
)
log = logging.getLogger("test_joradp")

TEST_YEAR   = 2025
TEST_ISSUES = range(1, 6)   # Issues 001 through 005

# ════════════════════════════════════════════════════════════════════════════
def run_test():
    log.info("=" * 65)
    log.info("[TEST] JORADP Pipeline Verification Test -- 2025 Issues 001-005")
    log.info("=" * 65)

    scraper   = JORADPScraper(sleep_seconds=2.0)
    extractor = PDFExtractor()
    parser    = GeminiParser()

    summary = {
        "year": TEST_YEAR,
        "issues_attempted": 0,
        "issues_downloaded": 0,
        "issues_skipped_404": 0,
        "total_pages_extracted": 0,
        "ocr_pages_used": 0,
        "total_articles_parsed": 0,
        "output_files": [],
    }

    all_articles: list[dict] = []   # collect for sample artifact

    for issue_num in TEST_ISSUES:
        summary["issues_attempted"] += 1
        log.info(f"\n-- Issue {issue_num:03d} --------------------------------------------------")

        # Stage 1: Download
        pdf_path = scraper.download(TEST_YEAR, issue_num)
        if pdf_path is None:
            summary["issues_skipped_404"] += 1
            log.warning(f"  [WARN] Issue {issue_num:03d} not available — skipping")
            continue

        summary["issues_downloaded"] += 1
        size_kb = pdf_path.stat().st_size // 1024
        log.info(f"  [PDF] PDF size: {size_kb} KB")

        # Stage 2: Extract
        extraction = extractor.extract(pdf_path)
        pages      = extraction["pages"]
        header     = extraction["header"]
        footer     = extraction["footer"]
        ocr_used   = extraction["ocr_used"]

        summary["total_pages_extracted"] += len(pages)
        if ocr_used:
            summary["ocr_pages_used"] += 1
            log.info("  [OCR] OCR fallback was used for one or more pages")

        log.info(f"  [PAGES] Pages extracted: {len(pages)}")
        log.info(f"  [HEADER] {header[:80].strip()!r}")

        # Stage 3: Parse with Gemini
        out_dir = OUTPUT_DIR / str(TEST_YEAR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"A{TEST_YEAR}{issue_num:03d}.jsonl"

        issue_articles: list[dict] = []

        with open(out_file, "w", encoding="utf-8") as f:
            for page_num, page_text in pages.items():
                articles = parser.parse_page(
                    page_text=page_text,
                    page_num=page_num,
                    header=header,
                    footer=footer,
                    year=TEST_YEAR,
                    issue=f"{issue_num:03d}",
                )
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
                    issue_articles.append(article)

        issue_count = len(issue_articles)
        summary["total_articles_parsed"] += issue_count
        summary["output_files"].append(str(out_file))
        all_articles.extend(issue_articles)

        log.info(f"  [OK] Parsed {issue_count} articles -> {out_file.name}")

    # ── Write sample artifact (first 10 articles) ──────────────────────────
    sample_path = Path(__file__).parent / "sample_joradp_output.jsonl"
    sample_articles = all_articles[:10]

    with open(sample_path, "w", encoding="utf-8") as f:
        for art in sample_articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")

    log.info(f"\n[SAMPLE] Sample artifact saved: {sample_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("[SUMMARY] TEST RESULTS")
    log.info("=" * 65)
    for key, val in summary.items():
        if key == "output_files":
            log.info(f"   {key}:")
            for f in val:
                log.info(f"      → {f}")
        else:
            log.info(f"   {key:<28}: {val}")

    # ── Schema validation spot check ──────────────────────────────────────
    log.info("\n[SCHEMA] Schema Spot-Check (first article):")
    if all_articles:
        first = all_articles[0]
        required_fields = [
            "id", "country", "law_domain", "law_name", "book", "title",
            "article_number", "classification", "text", "text_explanation",
            "summary", "legal_conditions_summary", "penalties_summary",
            "keywords", "legal_type", "norm_type", "relations", "source",
            "versioning",
        ]
        missing = [f for f in required_fields if f not in first]
        if missing:
            log.warning(f"   [WARN] Missing fields: {missing}")
        else:
            log.info("   [PASS] All required schema fields present")
        log.info(f"   Article ID   : {first.get('id', 'N/A')}")
        log.info(f"   Law name     : {first.get('law_name', 'N/A')}")
        log.info(f"   legal_type   : {first.get('legal_type', 'N/A')}")
        log.info(f"   norm_type    : {first.get('norm_type', 'N/A')}")
        preview = (first.get("text", {}).get("original", "") or "")[:120]
        log.info(f"   text.original: {preview}…")
    else:
        log.warning("   [WARN] No articles were parsed in this run.")

    log.info("\n[DONE] Test complete.")
    return summary


if __name__ == "__main__":
    run_test()
