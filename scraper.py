# -*- coding: utf-8 -*-
"""
scraper.py — JORADP Legal Data Ingestion Pipeline
===================================================
Algerian Official Gazette (الجريدة الرسمية) → Structured JSONL

Pipeline Stages:
  1. Download PDF issues from joradp.dz
  2. Extract text via pdfplumber (native) or pytesseract (OCR fallback for images)
  3. Parse articles with Gemini 1.5 Flash → structured JSON schema
  4. Write output to dataset/raw/joradp/{year}/*.jsonl

Usage:
  python scraper.py --year 2025 --issues 1-5
  python scraper.py --year 2025 --issues 1-88
  python scraper.py --years 1964-2025 --issues 1-999
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ── Logging setup (UTF-8 safe for Windows CP1252 console) ────────────────────
import io as _io
_safe_stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(_safe_stdout)],
)
log = logging.getLogger("joradp")

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("BM25_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    log.error("❌ No Gemini API key found. Set BM25_GEMINI_API_KEY in .env")
    sys.exit(1)

# ── Tesseract path (Windows — handles spaces via pytesseract config) ──────────
import pytesseract
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    TESSERACT_AVAILABLE = True
    log.info(f"[OK] Tesseract found at: {TESSERACT_PATH}")
else:
    TESSERACT_AVAILABLE = False
    log.warning("[WARN] Tesseract not found -- OCR fallback disabled (text-PDFs only)")

import pdfplumber

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    log.warning("⚠️  pdf2image not installed — OCR fallback disabled")

# ── Gemini client ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types as genai_types

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PDF_CACHE_DIR = BASE_DIR / "dataset" / "raw" / "joradp_pdfs"
OUTPUT_DIR    = BASE_DIR / "dataset" / "raw" / "joradp"

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Download
# ═════════════════════════════════════════════════════════════════════════════
class JORADPScraper:
    """Downloads Arabic-edition PDF issues from joradp.dz with retry and caching."""

    BASE_URL = "https://www.joradp.dz/FTP/jo-arabe/{year}/A{year}{issue:03d}.pdf"

    def __init__(self, sleep_seconds: float = 2.0):
        self.sleep_seconds = sleep_seconds
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        import urllib3
        # JORADP uses a certificate chain that Windows cannot verify natively.
        # We disable SSL verification and suppress the InsecureRequestWarning.
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        session = requests.Session()
        session.verify = False   # <-- SSL bypass for joradp.dz
        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            ),
            "Accept-Language": "ar,fr;q=0.9",
        })
        return session

    def build_url(self, year: int, issue: int) -> str:
        return self.BASE_URL.format(year=year, issue=issue)

    def download(self, year: int, issue: int) -> Optional[Path]:
        """Download PDF (cached). Returns local path or None on failure."""
        dest_dir = PDF_CACHE_DIR / str(year)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"A{year}{issue:03d}.pdf"

        if dest_file.exists() and dest_file.stat().st_size > 1024:
            log.info(f"  [CACHE] Cache hit: {dest_file.name}")
            return dest_file

        url = self.build_url(year, issue)
        log.info(f"  [DL] Downloading: {url}")
        try:
            resp = self.session.get(url, timeout=30, stream=True)
            if resp.status_code == 404:
                log.warning(f"  [SKIP] Issue not found (404): year={year} issue={issue:03d}")
                return None
            resp.raise_for_status()

            with open(dest_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            size_kb = dest_file.stat().st_size // 1024
            log.info(f"  [SAVED] {dest_file.name} ({size_kb} KB)")
            time.sleep(self.sleep_seconds)
            return dest_file

        except requests.RequestException as exc:
            log.error(f"  [ERROR] Download failed: {exc}")
            return None


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Text Extraction
# ═════════════════════════════════════════════════════════════════════════════
class PDFExtractor:
    """
    Extracts text from a PDF page by page.
    Primary: pdfplumber (fast, exact).
    Fallback: pdf2image + pytesseract (OCR for scanned images).
    """

    MIN_CHARS_FOR_TEXT = 50   # threshold: fewer chars → assume scanned page

    def extract(self, pdf_path: Path) -> dict:
        """
        Returns:
          {
            "pages": {1: "text...", 2: "text...", ...},
            "header": "first lines of page 1",
            "footer": "last lines of last page",
            "issue_year": str,
            "issue_num": str,
            "ocr_used": bool,
          }
        """
        result = {
            "pages": {},
            "header": "",
            "footer": "",
            "issue_year": self._extract_year_from_path(pdf_path),
            "issue_num": self._extract_issue_from_path(pdf_path),
            "ocr_used": False,
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()

                if len(text) < self.MIN_CHARS_FOR_TEXT and TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE:
                    # Fallback to OCR
                    text = self._ocr_page(pdf_path, page_num)
                    result["ocr_used"] = True

                result["pages"][page_num] = text

                # Collect header (first page, first 4 lines)
                if page_num == 1 and text:
                    lines = text.splitlines()
                    result["header"] = "\n".join(lines[:4])

        # Footer: last page, last 3 lines
        if result["pages"]:
            last_text = result["pages"][max(result["pages"].keys())]
            if last_text:
                lines = last_text.splitlines()
                result["footer"] = "\n".join(lines[-3:])

        total_pages = len(result["pages"])
        ocr_tag = " (OCR)" if result["ocr_used"] else ""
        log.info(f"  [TEXT] Extracted {total_pages} pages{ocr_tag}")
        return result

    def _ocr_page(self, pdf_path: Path, page_num: int) -> str:
        """Convert a single PDF page to image and run Tesseract (Arabic)."""
        try:
            images = convert_from_path(
                str(pdf_path),
                first_page=page_num,
                last_page=page_num,
                dpi=300,
            )
            if not images:
                return ""
            text = pytesseract.image_to_string(images[0], lang="ara")
            return text.strip()
        except Exception as exc:
            log.warning(f"  ⚠️  OCR failed on page {page_num}: {exc}")
            return ""

    @staticmethod
    def _extract_year_from_path(pdf_path: Path) -> str:
        # Path: .../joradp_pdfs/2025/A2025001.pdf
        return pdf_path.parent.name

    @staticmethod
    def _extract_issue_from_path(pdf_path: Path) -> str:
        # filename: A2025001.pdf → "001"
        match = re.search(r"A\d{4}(\d{3})", pdf_path.stem)
        return match.group(1) if match else "000"


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Gemini Parsing
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """أنت محلل قانوني متخصص في القانون الجزائري ومهمتك استخراج المواد القانونية من الجريدة الرسمية الجزائرية.

قواعد الاستخراج:
1. استخرج كل مادة قانونية على حِدة كـ JSON object منفصل.
2. حدّد حقل "book" و"title" من ترويسات الصفحة (رأس الصفحة / ذيلها) التي سيُزوَّد بها.
3. لحقل "legal_type": استنتجه من طبيعة المادة:
   - "مدنية / موضوعية" للمواد التي تنشئ أو تعدّل حقوقاً.
   - "إجرائية" للمواد المتعلقة بالإجراءات والمهل.
   - "جزائية" للمواد التي تتضمن عقوبات.
   - "تنظيمية / إدارية" للمواد التنظيمية.
4. لحقل "norm_type": استنتجه من الأسلوب اللغوي:
   - "آمرة" إذا احتوت على: يسري، يجب، يلتزم، يُحظر، لا يجوز.
   - "مكملة" إذا احتوت على: يجوز، يحق، يمكن (قابلة للاستبعاد الاتفاقي).
   - "تخييرية" إذا منحت خياراً صريحاً.
5. لحقل "text_explanation": قدّم تعليقاً قانونياً مختصراً يشرح التطبيق العملي للمادة بالعربية (3-4 جمل).
6. أعِد JSON فقط — بدون أي نص قبله أو بعده.
7. إذا كانت الصفحة تحتوي على مواد متعددة، أعِد مصفوفة JSON Array.
8. إذا لم تجد مواد قانونية واضحة في النص، أعِد مصفوفة فارغة: []
"""

ARTICLE_PROMPT_TEMPLATE = """سياق الجريدة الرسمية:
- السنة: {year}
- العدد: {issue}
- رأس الصفحة: {header}
- ذيل الصفحة: {footer}
- رقم الصفحة: {page_num}

نص الصفحة:
{page_text}

---
استخرج كل المواد القانونية الواردة في هذا النص وأعِد النتيجة كـ JSON Array.
لكل مادة، استخدم هذا الـ Schema بالضبط:
{{
  "id": "DZ_JO_{year}_{issue}_ART_{{رقم_المادة}}",
  "country": "Algeria",
  "law_domain": "استنتجه من السياق (مثال: Civil Law / Penal Law / Administrative Law)",
  "law_name": "اسم القانون أو المرسوم كاملاً من الترويسة",
  "book": "الكتاب والباب المستخرج من ترويسة الصفحة",
  "title": "عنوان الفصل أو القسم",
  "article_number": رقم_صحيح_أو_نص_مثل_"17_مكرر",
  "classification": {{
    "main_category": "الفئة الرئيسية",
    "sub_category": "الفئة الفرعية",
    "section": "القسم"
  }},
  "text": {{
    "original": "النص الحرفي للمادة بالعربية",
    "language": "ar"
  }},
  "text_explanation": "تعليق قانوني مختصر على التطبيق العملي للمادة (3-4 جمل بالعربية)",
  "summary": "ملخص من جملة واحدة للمادة",
  "legal_conditions_summary": "الشروط القانونية اللازمة لتطبيق المادة",
  "penalties_summary": "ملخص العقوبات إن وُجدت، وإلا: 'غير متاح'",
  "keywords": ["كلمة مفتاحية 1", "كلمة مفتاحية 2", "..."],
  "legal_type": "مدنية / موضوعية | إجرائية | جزائية | تنظيمية / إدارية",
  "norm_type": "آمرة | مكملة | تخييرية",
  "relations": {{
    "related_articles": [],
    "derived_from": null,
    "amended_by": null,
    "repeals": null
  }},
  "source": {{
    "document": "الجريدة الرسمية العدد {issue} لسنة {year}",
    "year": {year_int},
    "publisher": "الجريدة الرسمية للجمهورية الجزائرية الديمقراطية الشعبية"
  }},
  "versioning": {{
    "status": "ساري",
    "version": "v1.0",
    "last_update": "{today}"
  }}
}}
"""


class GeminiParser:
    """Sends page text to Gemini 1.5 Flash and returns a list of article dicts."""

    MODEL = "gemini-1.5-flash"
    RPM_DELAY = 4.5   # seconds between requests to stay under 15 RPM

    def parse_page(
        self,
        page_text: str,
        page_num: int,
        header: str,
        footer: str,
        year: int,
        issue: str,
    ) -> list[dict]:
        """Returns list of article dicts, may be empty."""
        if len(page_text.strip()) < 30:
            return []

        prompt = ARTICLE_PROMPT_TEMPLATE.format(
            year=year,
            issue=issue,
            header=header or "غير متوفر",
            footer=footer or "غير متوفر",
            page_num=page_num,
            page_text=page_text[:4000],   # cap at ~4k chars to stay within context
            year_int=year,
            today=date.today().isoformat(),
        )

        try:
            response = _gemini_client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            raw = response.text.strip()
            articles = self._parse_json(raw)
            time.sleep(self.RPM_DELAY)
            return articles

        except Exception as exc:
            log.error(f"  ❌ Gemini error on page {page_num}: {exc}")
            time.sleep(self.RPM_DELAY)
            return []

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        """Strip markdown fences and parse JSON array."""
        # Remove ```json ... ``` fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw or raw == "[]":
            return []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            return [a for a in parsed if isinstance(a, dict)]
        except json.JSONDecodeError as exc:
            log.warning(f"  ⚠️  JSON parse error: {exc} — raw snippet: {raw[:200]}")
            return []


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(years: list[int], issue_range: tuple[int, int], dry_run: bool = False):
    """
    Main pipeline: for each (year, issue), download → extract → parse → write JSONL.
    """
    scraper  = JORADPScraper(sleep_seconds=2.0)
    extractor = PDFExtractor()
    parser   = GeminiParser()

    total_articles = 0
    total_issues   = 0
    skipped_issues = 0

    for year in years:
        year_out_dir = OUTPUT_DIR / str(year)
        year_out_dir.mkdir(parents=True, exist_ok=True)

        for issue_num in range(issue_range[0], issue_range[1] + 1):
            log.info(f"\n{'='*60}")
            log.info(f"📰  Processing: Year={year}  Issue={issue_num:03d}")
            log.info(f"{'='*60}")

            # ── Stage 1: Download ──────────────────────────────────────
            pdf_path = scraper.download(year, issue_num)
            if pdf_path is None:
                skipped_issues += 1
                continue

            if dry_run:
                log.info("  [DRY RUN] Skipping extraction and parsing.")
                total_issues += 1
                continue

            # ── Stage 2: Extract ───────────────────────────────────────
            extraction = extractor.extract(pdf_path)
            header = extraction["header"]
            footer = extraction["footer"]
            pages  = extraction["pages"]

            # ── Stage 3: Parse ─────────────────────────────────────────
            output_file = year_out_dir / f"A{year}{issue_num:03d}.jsonl"
            issue_articles = 0

            with open(output_file, "w", encoding="utf-8") as out_f:
                for page_num, page_text in pages.items():
                    articles = parser.parse_page(
                        page_text=page_text,
                        page_num=page_num,
                        header=header,
                        footer=footer,
                        year=year,
                        issue=f"{issue_num:03d}",
                    )

                    for article in articles:
                        # Ensure id uniqueness
                        if not article.get("id"):
                            article["id"] = f"DZ_JO_{year}_{issue_num:03d}_ART_{uuid.uuid4().hex[:8]}"

                        out_f.write(json.dumps(article, ensure_ascii=False) + "\n")
                        issue_articles += 1

            log.info(f"  ✅ Issue {issue_num:03d}: {issue_articles} articles → {output_file.name}")
            total_articles += issue_articles
            total_issues   += 1

    log.info(f"\n{'='*60}")
    log.info(f"🏁  PIPELINE COMPLETE")
    log.info(f"   Issues processed : {total_issues}")
    log.info(f"   Issues skipped   : {skipped_issues}")
    log.info(f"   Total articles   : {total_articles}")
    log.info(f"   Output directory : {OUTPUT_DIR}")
    log.info(f"{'='*60}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def parse_issue_range(s: str) -> tuple[int, int]:
    """Parse '1-5' → (1,5) or '7' → (7,7)."""
    if "-" in s:
        start, end = s.split("-", 1)
        return int(start.strip()), int(end.strip())
    n = int(s.strip())
    return n, n


def parse_year_range(s: str) -> list[int]:
    """Parse '2025' → [2025] or '2020-2025' → [2020,2021,...,2025]."""
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start.strip()), int(end.strip()) + 1))
    return [int(s.strip())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JORADP Legal Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py --year 2025 --issues 1-5
  python scraper.py --year 2025 --issues 1-88
  python scraper.py --years 1964-2025 --issues 1-999
  python scraper.py --year 2025 --issues 1-3 --dry-run
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year",  type=str, help="Single year, e.g. 2025")
    group.add_argument("--years", type=str, help="Year range, e.g. 2020-2025")

    parser.add_argument(
        "--issues", type=str, default="1-5",
        help="Issue range, e.g. '1-5' or '42' (default: 1-5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download PDFs only, skip extraction and Gemini parsing",
    )
    args = parser.parse_args()

    years_list   = parse_year_range(args.year or args.years)
    issues_tuple = parse_issue_range(args.issues)

    log.info(f"🚀  Starting JORADP pipeline")
    log.info(f"   Years  : {years_list}")
    log.info(f"   Issues : {issues_tuple[0]}–{issues_tuple[1]}")
    log.info(f"   Dry run: {args.dry_run}")

    run_pipeline(years_list, issues_tuple, dry_run=args.dry_run)
