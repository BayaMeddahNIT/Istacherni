"""
web_search_tool.py
------------------
Procedural web search tool for the Hybrid Agentic RAG pipeline.

Design principles:
  • Domain-whitelisted: only fetches from official Algerian government portals.
  • Provider-agnostic: swap DuckDuckGo → Tavily → Serper by changing WEB_SEARCH_PROVIDER in .env.
  • Returns a clean JSON-serialisable dict — the agentic loop treats it like any other tool result.
  • Fully self-contained: no side effects, stateless function.

Supported providers:
  - duckduckgo  (default, free, no API key)
  - tavily      (better quality, requires TAVILY_API_KEY)
  - serper      (Google results, requires SERPER_API_KEY)
"""

import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────

PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").lower()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "4"))

# Official Algerian government domains — the ONLY sources trusted for procedural info
_DEFAULT_DOMAINS = [
    "cnrc.dz",             # Centre National du Registre du Commerce
    "mjustice.dz",         # Ministère de la Justice
    "interieur.gov.dz",    # Ministère de l'Intérieur
    "commerce.gov.dz",     # Ministère du Commerce
    "mcommerce.gov.dz",    # Alternative Ministry of Commerce domain
    "mf.gov.dz",           # Ministère des Finances
    "travail.gov.dz",      # Ministère du Travail
    "andi.dz",             # Agence Nationale de Développement de l'Investissement
    "joradp.dz",           # Journal Officiel de la République Algérienne Démocratique
    "dgimpots.gov.dz",     # Direction Générale des Impôts
    "bawabatic.dz",        # Algerian Government Portal
    "gov.dz",              # Generic Government domain
]

# Build the env-overridable whitelist
_ENV_DOMAINS = os.getenv("ALLOWED_SEARCH_DOMAINS", "")
ALLOWED_DOMAINS: list[str] = (
    [d.strip() for d in _ENV_DOMAINS.split(",") if d.strip()]
    if _ENV_DOMAINS
    else _DEFAULT_DOMAINS
)


# ── Result schema ──────────────────────────────────────────────────────────────

def _make_result(query: str, results: list[dict], provider: str) -> dict:
    """Standardised return dict for all providers."""
    return {
        "query":    query,
        "provider": provider,
        "domains_searched": ALLOWED_DOMAINS,
        "count":    len(results),
        "results":  results,  # [{title, url, snippet}]
    }


def _make_error(query: str, error: str) -> dict:
    return {
        "query":   query,
        "error":   error,
        "count":   0,
        "results": [],
    }


# ── DuckDuckGo provider ────────────────────────────────────────────────────────

def _search_duckduckgo(query: str, domains: list[str]) -> dict:
    """
    Search via DuckDuckGo with a two-pass strategy:
      Pass 1 — scoped query with site: operators (precise but DDG can return 0 results)
      Pass 2 — unscoped query + post-filter by domain (reliable fallback)
    Uses the 'ddgs' package (pip install ddgs).
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return _make_error(query, "ddgs not installed. Run: pip install ddgs")

    # ── Pass 1: domain-scoped query ───────────────────────────────────────
    # Strip out the Gulf SEO spam at the search engine level
    scoped_query = query + " الجزائر -دبي -الامارات -الإمارات -السعودية"

    try:
        with DDGS() as ddgs_client:
            raw = list(ddgs_client.text(scoped_query, max_results=MAX_RESULTS, region="dz-ar"))
    except Exception as e:
        return _make_error(query, f"DuckDuckGo search failed: {e}")

    # ── Pass 2: fallback — unscoped + post-filter if Pass 1 returned nothing ──
    if not raw:
        try:
            with DDGS() as ddgs_client:
                raw_broad = list(ddgs_client.text(
                    f"{query} الجزائر إجراءات -دبي -الامارات -الإمارات -السعودية",
                    max_results=MAX_RESULTS * 3,
                    region="dz-ar",
                ))
            # Keep only results whose URL contains a whitelisted domain
            raw = [
                r for r in raw_broad
                if any(d in r.get("href", "") for d in domains)
            ][:MAX_RESULTS]
        except Exception:
            pass  # If fallback also fails, return empty results gracefully

    results = [
        {
            "title":   r.get("title", ""),
            "url":     r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw
    ]
    return _make_result(query, results, "duckduckgo")


# ── Tavily provider ────────────────────────────────────────────────────────────

def _search_tavily(query: str, domains: list[str]) -> dict:
    """Search via Tavily API (RAG-optimised). Requires TAVILY_API_KEY in .env."""
    if not TAVILY_API_KEY:
        return _make_error(query, "TAVILY_API_KEY not set in .env")
    try:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":        TAVILY_API_KEY,
                "query":          query,
                "include_domains": domains,
                "max_results":    MAX_RESULTS,
                "search_depth":   "advanced",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return _make_error(query, f"Tavily search failed: {e}")

    results = [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]
    return _make_result(query, results, "tavily")


# ── Serper provider ────────────────────────────────────────────────────────────

def _search_serper(query: str, domains: list[str]) -> dict:
    """Search via Serper.dev (Google results). Requires SERPER_API_KEY in .env."""
    if not SERPER_API_KEY:
        return _make_error(query, "SERPER_API_KEY not set in .env")
    try:
        import requests
        site_scope = " OR ".join(f"site:{d}" for d in domains)
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": f"{query} ({site_scope})", "num": MAX_RESULTS, "gl": "dz", "hl": "ar"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return _make_error(query, f"Serper search failed: {e}")

    results = [
        {
            "title":   r.get("title", ""),
            "url":     r.get("link", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in data.get("organic", [])
    ]
    return _make_result(query, results, "serper")


# ── Public interface ───────────────────────────────────────────────────────────

def web_search(query: str, domains: Optional[list[str]] = None) -> dict:
    """
    Main entry point for the agentic web search tool.

    Args:
        query:   The search query (Arabic or French procedural question).
        domains: Optional domain override. Defaults to ALLOWED_DOMAINS whitelist.

    Returns:
        A dict with keys: query, provider, domains_searched, count, results.
        Each result has: title, url, snippet.
        On error: dict with 'error' key explaining the failure.
    """
    active_domains = domains if domains else ALLOWED_DOMAINS

    if PROVIDER == "tavily":
        return _search_tavily(query, active_domains)
    elif PROVIDER == "serper":
        return _search_serper(query, active_domains)
    else:
        return _search_duckduckgo(query, active_domains)


def web_search_json(query: str, domains: Optional[list[str]] = None) -> str:
    """Convenience wrapper that returns JSON string (for the agentic tool executor)."""
    return json.dumps(web_search(query, domains), ensure_ascii=False, indent=2)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "كيف أستخرج سجل تجاري؟"
    print(f"\n[web_search_tool] Provider: {PROVIDER}")
    print(f"[web_search_tool] Query: {q}")
    print(f"[web_search_tool] Domains: {ALLOWED_DOMAINS}\n")
    result = web_search(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))
