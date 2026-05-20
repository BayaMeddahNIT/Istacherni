"""
jwt_handler.py
--------------
JWT token creation, verification, and the FastAPI dependency used by protected routes.

Token structure:
  {
    "sub":  "<user_id>",
    "username": "<username>",
    "exp":  <unix timestamp>,
    "type": "access" | "refresh"
  }
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

# ── Config ────────────────────────────────────────────────────────────────────
# Override via environment variable in production.
SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "ISTACHERNI_DEV_SECRET_CHANGE_IN_PROD_!!!")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = 60 * 24        # 24 hours
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7   # 7 days

_bearer_scheme = HTTPBearer(auto_error=True)

# ── Token creation ─────────────────────────────────────────────────────────────

def _make_token(data: dict, expires_delta: timedelta, token_type: str) -> str:
    payload = data.copy()
    payload["exp"]  = datetime.now(timezone.utc) + expires_delta
    payload["type"] = token_type
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str, username: str) -> str:
    return _make_token(
        {"sub": user_id, "username": username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "access",
    )


def create_refresh_token(user_id: str, username: str) -> str:
    return _make_token(
        {"sub": user_id, "username": username},
        timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
        "refresh",
    )

# ── Token verification ─────────────────────────────────────────────────────────

def _decode_token(token: str, expected_type: str = "access") -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise credentials_exception

    if payload.get("type") != expected_type:
        raise credentials_exception
    if not payload.get("sub"):
        raise credentials_exception

    return payload


def verify_refresh_token(token: str) -> dict:
    return _decode_token(token, expected_type="refresh")


def verify_access_token(token: str) -> dict:
    """Verify an access token string directly (used for query-param auth on SSE)."""
    return _decode_token(token, expected_type="access")


# ── FastAPI dependency ─────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency. Inject into any route that requires authentication.
    Returns the decoded JWT payload dict with at least: {"sub": ..., "username": ...}
    """
    return _decode_token(credentials.credentials, expected_type="access")


# ── SSE-safe dependency (Bearer header OR ?token= query param) ────────────────
from fastapi import Query

def get_current_user_flexible(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    token: Optional[str] = Query(default=None, description="JWT token for SSE clients"),
) -> dict:
    """
    Auth dependency for streaming endpoints.
    EventSource/SSE clients cannot set headers, so we accept the JWT as either:
      - Authorization: Bearer <token>  (standard, used by apiFetch)
      - ?token=<token>                 (fallback for EventSource)
    """
    raw: Optional[str] = None
    if credentials:
        raw = credentials.credentials
    elif token:
        raw = token

    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(raw, expected_type="access")

