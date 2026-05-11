"""
rate_limiter.py
---------------
SlowAPI rate-limiter configuration.

Limits:
  /auth/*                 → 20 req / minute  (per IP)
  /chat                   → 10 req / minute  (per IP)
  /api/chat/stream        → 3  req / minute  (per IP)
  /search                 → 20 req / minute  (per IP)
  /sessions/*             → 30 req / minute  (per IP)

Usage in app.py:
  from backend.middleware.rate_limiter import limiter, rate_limit_handler
  app.state.limiter = limiter
  app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

  @app.post("/chat")
  @limiter.limit("10/minute")
  def chat(request: Request, ...): ...
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Key function: identify clients by their remote IP
limiter = Limiter(key_func=get_remote_address)


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a clean 429 JSON response instead of slowapi's default HTML page."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}. Please slow down.",
        },
        headers={"Retry-After": "60"},
    )
