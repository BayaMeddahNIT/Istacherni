"""
logging.py
----------
Structured request/response logging middleware.

Each handled request appends a single JSON line to:
  evaluation/api_logs.jsonl

Schema per entry:
  {
    "timestamp":     ISO-8601 string,
    "session_id":    str | null,
    "user_id":       str | null,
    "query":         str,
    "rag_type":      str,
    "retrieved_ids": [str],
    "latency_ms":    int,
    "status_code":   int,
    "error":         str | null
  }
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

# ── Log file path ─────────────────────────────────────────────────────────────
_LOG_DIR  = Path(__file__).resolve().parents[2] / "evaluation"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "api_logs.jsonl"


def log_request(
    *,
    query: str,
    rag_type: str,
    retrieved_ids: list[str],
    latency_ms: int,
    status_code: int,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Append a structured log entry (fire-and-forget, never raises)."""
    entry: dict[str, Any] = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id":    session_id,
        "user_id":       user_id,
        "query":         query[:500],
        "rag_type":      rag_type,
        "retrieved_ids": retrieved_ids[:20],
        "latency_ms":    latency_ms,
        "status_code":   status_code,
        "error":         error,
    }
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never crash the request
