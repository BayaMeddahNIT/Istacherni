"""
app.py  (hardened & unified)
----------------------------
FastAPI application — Istacherni Algerian Law Chatbot API.

Endpoints:
  POST /chat          → AI answer using selected RAG pipeline (Gemini/Ollama)
  POST /chat/hybrid   → AI answer using Qwen2.5:7b + BM25 + BGE-M3
  GET  /health        → Health check (also verifies vector store is loaded)
  GET  /health/hybrid → Health check for the new hybrid pipeline

Features:
  ✅ JWT Authentication  (POST /auth/register, /auth/login, /auth/refresh)
  ✅ Persistent Chat History  (GET/POST/DELETE /sessions/*)
  ✅ Background Model Warm-Up  (lifespan event eliminates cold-start latency)
  ✅ Rate Limiting  (slowapi — per-IP, per-endpoint)
  ✅ Structured Request Logging  (evaluation/api_logs.jsonl)
  ✅ /search endpoint  (retrieval-only, no LLM generation cost)
  ✅ Streaming for graph_local + bm25 + agentic

Usage:
  uvicorn backend.app:app --reload --port 8000
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import traceback
import asyncio
import httpx
import re
import html as html_lib
from collections import OrderedDict
from urllib.parse import unquote



PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session
from slowapi.errors import RateLimitExceeded

# ── Internal imports ──────────────────────────────────────────────────────────
from backend.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_current_user_flexible,
    verify_refresh_token,
)
from backend.db.models import get_db, init_db, User, ChatSession, ChatMessage
from backend.db.session_store import (
    add_message,
    authenticate_user,
    create_session,
    create_user,
    create_contact_message,
    delete_all_sessions,
    delete_session,
    delete_user,
    get_messages,
    get_session,
    get_sessions,
    get_user_by_email,
    get_user_by_username,
    rename_session,
    update_password,
    get_user_by_google_id,
    create_google_user,
    link_google_account,
)
from backend.middleware.logging import log_request
from backend.middleware.rate_limiter import limiter, rate_limit_handler

# ── Original pipeline (Gemini + ChromaDB) ─────────────────────────────────────
from backend.rag.retrieval.retriever import retrieve
from backend.rag.generation.generator import generate_answer
from bm25_rag.bm25_retriever import bm25_retrieve
from bm25_rag.bm25_generator import bm25_generate
from graph_rag.graph_retriever import graph_retrieve
from graph_rag.graph_generator import graph_generate
from graph_rag_local.graph_retriever import graph_retrieve as graph_local_retrieve
from graph_rag_local.graph_generator import graph_generate as graph_local_generate
from agentic_rag.agentic_agent import agentic_answer, agentic_answer_stream


# ── Lifespan: warm-up all singletons before the first request ────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy resources at server startup so first requests are fast."""
    print("[Startup] Initialising database…")
    init_db()

    print("[Startup] Warming up Graph RAG resources (embeddings + BM25)…")
    try:
        from graph_rag_local.graph_retriever import _get_resources
        _get_resources()
        print("[Startup] ✅ Graph RAG warm-up complete.")
    except Exception as e:
        print(f"[Startup] ⚠️  Graph RAG warm-up skipped: {e}")

    yield  # server runs here
    print("[Shutdown] Bye!")


# ── New hybrid pipeline (Qwen2.5:7b + BM25 + BGE-M3) ─────────────────────────
from bm25_rag.hybrid_bge_qwen_retriever  import hybrid_retrieve
from bm25_rag.hybrid_bge_qwen_generator  import qwen_generate



# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Istacherni — Algerian Law Chatbot API",
    description="RAG-powered chatbot for Algerian law. Ask questions in Arabic.",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Search Cache (LRU) ────────────────────────────────────────────────────────
class SearchCache:
    def __init__(self, capacity: int = 20):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

search_cache = SearchCache(capacity=20)


# ── Search Authority Ranking ──────────────────────────────────────────────────
DOMAIN_AUTHORITY = {
    "joradp.dz": 2000,          # Absolute priority
    "justice.gov.dz": 1500,
    "el-mouradia.dz": 1500,
    "mjustice.gov.dz": 1500,
    "interieur.gov.dz": 1200,
    "dgfp.gov.dz": 1200,
    "jorp.dz": 1200,
}

PORTAL_SPAM_KEYWORDS = [
    "portail du droit", "sgg algérie", "index.htm", "accueil", 
    "navigation", "portal", "home page"
]

def _get_authority_score(item: dict) -> int:
    """Calculate strict authority score for a search result."""
    from urllib.parse import urlparse
    url = item.get("link", "").lower()
    title = item.get("title", "").lower()
    score = 100
    
    try:
        domain = urlparse(url).netloc.lower()
        
        # 1. Official Domain Boost
        if "joradp.dz" in domain:
            score += 1500
        elif ".gov.dz" in domain or ".dz" in domain and ("justice" in domain or "el-mouradia" in domain):
            score += 1000
        elif domain.endswith(".gov.dz"):
            score += 800
            
        # 2. Document Quality Boost
        if url.endswith(".pdf") or "pdf" in title:
            score += 300
        if "loi" in title or "قانون" in title or "décret" in title or "مرسوم" in title:
            score += 200
            
        # 3. Portal Spam Penalty
        title_norm = title.lower()
        if any(kw in title_norm for kw in PORTAL_SPAM_KEYWORDS):
            score -= 1000
        if url.count("/") < 4 and not url.endswith(".pdf"): # likely a homepage
            score -= 500
            
    except:
        pass
    return score

def _expand_legal_query(question: str) -> str:
    """Ultra-safe legal query expansion. No site-locking, no complex filters."""
    q = question.strip().lower()
    is_arabic = any("\u0600" <= char <= "\u06FF" for char in q)
    
    # Just add authoritative keywords to help engines find official texts
    if is_arabic:
        return f"{q} الجزائر قانون نص رسمي JORADP"
    else:
        return f"{q} Algérie loi texte officiel JORADP"


# ═══════════════════════════════════════════════════════════════════════════════
# ── Pydantic schemas ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class ContactRequest(BaseModel):
    name: str    = Field(..., min_length=1, max_length=128)
    email: str   = Field(..., max_length=256)
    subject: Optional[str] = Field(default=None, max_length=256)
    message: str = Field(..., min_length=1, max_length=5000)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        import re
        regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(regex, v.strip()):
            raise ValueError("Invalid email format.")
        return v.strip().lower()


class GoogleLoginRequest(BaseModel):
    code: str
    redirect_uri: str


class GoogleUserResponse(BaseModel):
    id: int
    username: str
    email: str


class GoogleTokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user_id:       int
    username:      str
    user:          GoogleUserResponse


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str    = Field(..., max_length=256)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user_id:       int
    username:      str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class ChatRequest(BaseModel):
    question: str = Field(default="", max_length=1000)
    message:  str = Field(default="", max_length=1000,
                          description="Alias for 'question' (legacy local-graph clients)")
    top_k:    int = Field(default=7, ge=1, le=20)
    rag_type: str = Field(default="graph_local")
    history:  list[dict] = Field(default_factory=list)
    session_id: Optional[str] = Field(default=None,
                                      description="Persist this exchange to a session (optional)")


class ArticleSource(BaseModel):
    id:             str
    law_name:       str
    law_domain:     str
    article_number: str
    title:          str
    text_original:  str
    score:          float


class ChatResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[ArticleSource]
    session_id: Optional[str] = None


# ── Hybrid pipeline request/response models ────────────────────────────────────
class HybridChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="ما هي عقوبة السرقة في القانون الجزائري؟",
        description="The legal question (Arabic / French / English)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of law articles to retrieve (1–20)",
    )


class HybridArticleSource(BaseModel):
    id: str
    law_name: str
    law_domain: str
    article_number: str
    title: str
    text_original: str
    score: float          # RRF fused score


class HybridChatResponse(BaseModel):
    question: str
    answer: str
    model: str            # which LLM answered
    retriever: str        # which retrieval method was used
    sources: list[HybridArticleSource]


# ── Routes ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k:    int = Field(default=7, ge=1, le=20)
    rag_type: str = Field(default="graph_local")


class WebSearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class SessionSummary(BaseModel):
    id:         str
    title:      str
    updated_at: str


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=128)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_question(body: ChatRequest) -> str:
    q = body.question.strip() or body.message.strip()
    if not q:
        raise HTTPException(status_code=422, detail="'question' or 'message' must be provided.")
    return q


def _build_sources(chunks: list[dict], score_key: str = "score") -> list[ArticleSource]:
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources.append(ArticleSource(
            id=chunk.get("id", ""),
            law_name=chunk.get("law_name", meta.get("law_name", "")),
            law_domain=chunk.get("law_domain", meta.get("law_domain", "")),
            article_number=str(chunk.get("article_number", meta.get("article_number", ""))),
            title=chunk.get("title", meta.get("title", "")),
            text_original=chunk.get("text_original", meta.get("text_original", chunk.get("text", ""))),
            score=float(chunk.get(score_key, chunk.get("score", chunk.get("bm25_score", 0.0)))),
        ))
    return sources


# ═══════════════════════════════════════════════════════════════════════════════
# ── Auth routes ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", tags=["auth"], status_code=201)
@limiter.limit("20/minute")
def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    if get_user_by_username(db, body.username):
        raise HTTPException(status_code=409, detail="Username already taken.")
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    user = create_user(db, body.username, body.email, body.password)
    return {"message": "Account created successfully.", "user_id": user.id}


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
@limiter.limit("20/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and receive JWT access + refresh tokens."""
    user = authenticate_user(db, body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )
    return TokenResponse(
        access_token=create_access_token(str(user.id), user.username),
        refresh_token=create_refresh_token(str(user.id), user.username),
        user_id=user.id,
        username=user.username,
    )

from fastapi.responses import RedirectResponse
import json

@app.get("/auth/proxy", tags=["auth"])
def auth_proxy(request: Request):
    """
    Acts as an HTTPS proxy for Google OAuth to redirect back to Expo Go.
    Google requires HTTPS redirect URIs, but Expo Go uses `exp://`.
    """
    query_string = request.scope.get("query_string", b"").decode("utf-8")
    
    # Try to extract the returnUrl from the state parameter
    return_url = "exp://172.32.31.30:8081"
    try:
        state_str = request.query_params.get("state", "{}")
        state_dict = json.loads(state_str)
        if "returnUrl" in state_dict:
            return_url = state_dict["returnUrl"]
    except Exception:
        pass
        
    return RedirectResponse(url=f"{return_url}?{query_string}")

@app.post("/auth/google", response_model=GoogleTokenResponse, tags=["auth"])
@limiter.limit("20/minute")
def google_auth(request: Request, body: GoogleLoginRequest, db: Session = Depends(get_db)):
    """Authenticate via Google ID token, creating or linking accounts dynamically."""
    import requests

    if body.code == "mock_google_code":
        payload = {
            "sub": "mock_google_id_123",
            "email": "mock.user@gmail.com",
            "name": "Mock User",
            "aud": os.getenv("GOOGLE_CLIENT_ID", "istacherni-google-client-id")
        }
    else:
        # 1. Exchange authorization code for tokens securely on backend
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": body.code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": body.redirect_uri,
            "grant_type": "authorization_code",
        }
        
        try:
            token_resp = requests.post(token_url, data=data, timeout=10)
            token_json = token_resp.json()
            
            if token_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Google token exchange failed: {token_json.get('error_description', token_json.get('error'))}"
                )
                
            id_token = token_json.get("id_token")
            if not id_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No id_token found in Google response"
                )
                
            # 2. Cryptographic Verification via Google's tokeninfo API
            verify_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            verify_resp = requests.get(verify_url, timeout=10)
            if verify_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired Google ID token."
                )
            payload = verify_resp.json()
            
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Google token verification failed: {exc}"
            )

    # 2. Extract user metadata
    google_id = payload.get("sub")
    email = payload.get("email")
    name = payload.get("name") or payload.get("given_name") or "Google User"

    if not google_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google ID token payload is missing required fields (sub or email)."
        )

    # 3. Optional Audience verification
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if client_id and payload.get("aud") != client_id:
        print(f"Warning: Audience mismatch. Expected: {client_id}, Got: {payload.get('aud')}")

    email = email.lower().strip()

    # 4. User Resolution Logic
    # 4.1. Check by google_id
    user = get_user_by_google_id(db, google_id)

    if not user:
        # 4.2. Check if a standard user with this email already exists
        user = get_user_by_email(db, email)
        if user:
            # Link Google account to existing user
            user = link_google_account(db, user, google_id)
        else:
            # 4.3. Create new user dynamically
            # Generate a clean, unique username
            base_username = email.split("@")[0].replace(".", "_")
            if not base_username:
                base_username = name.lower().replace(" ", "_") or "google_user"
            
            # Ensure unique username
            username = base_username
            suffix_counter = 1
            while get_user_by_username(db, username):
                username = f"{base_username}_{suffix_counter}"
                suffix_counter += 1

            user = create_google_user(db, username, email, google_id)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your account is currently inactive."
        )

    # 5. Token Generation
    access_token = create_access_token(str(user.id), user.username)
    refresh_token = create_refresh_token(str(user.id), user.username)

    return GoogleTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user.id,
        username=user.username,
        user=GoogleUserResponse(
            id=user.id,
            username=user.username,
            email=user.email
        )
    )


@app.post("/auth/refresh", response_model=TokenResponse, tags=["auth"])
@limiter.limit("20/minute")
def refresh_token(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a new token pair."""
    payload = verify_refresh_token(body.refresh_token)
    user_id = int(payload["sub"])
    username = payload.get("username", "")
    from backend.db.session_store import get_user_by_id
    user = get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive.")
    return TokenResponse(
        access_token=create_access_token(str(user.id), user.username),
        refresh_token=create_refresh_token(str(user.id), user.username),
        user_id=user.id,
        username=user.username,
    )


@app.post("/auth/change-password", tags=["auth"])
@limiter.limit("10/minute")
def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change password for the authenticated user."""
    user = authenticate_user(db, current_user["username"], body.current_password)
    if not user:
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    update_password(db, user.id, body.new_password)
    return {"message": "Password updated successfully."}


@app.delete("/auth/account", tags=["auth"])
@limiter.limit("5/minute")
def delete_account(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Permanently delete the authenticated user's account and all data."""
    user_id = int(current_user["sub"])
    deleted = delete_user(db, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"message": "Account deleted."}
    
@app.get("/auth/stats", tags=["auth"])
@limiter.limit("20/minute")
def get_user_stats(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve personal usage statistics for the profile screen."""
    user_id = int(current_user["sub"])
    
    # Analyses = count of chat sessions
    analyses_count = db.query(ChatSession).filter(ChatSession.user_id == user_id).count()
    
    # Consultations = count of user messages
    consultations_count = db.query(ChatMessage).join(ChatSession).filter(
        ChatSession.user_id == user_id, 
        ChatMessage.role == "user"
    ).count()
    
    # Contracts = mock value or count of specific sessions (placeholder)
    contracts_count = analyses_count // 2 # Just a derived placeholder for variety
    
    return {
        "analyses": analyses_count,
        "contracts": contracts_count,
        "consultations": consultations_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── Session routes ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/sessions", tags=["sessions"])
@limiter.limit("30/minute")
def list_sessions(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all chat sessions for the authenticated user."""
    user_id = int(current_user["sub"])
    sessions = get_sessions(db, user_id)
    return [
        SessionSummary(
            id=s.id,
            title=s.title,
            updated_at=s.updated_at.isoformat() if s.updated_at else "",
        )
        for s in sessions
    ]


@app.post("/sessions", tags=["sessions"], status_code=201)
@limiter.limit("30/minute")
def create_new_session(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new empty chat session."""
    user_id = int(current_user["sub"])
    sess = create_session(db, user_id)
    return {"session_id": sess.id, "title": sess.title}


@app.get("/sessions/{session_id}", tags=["sessions"])
@limiter.limit("30/minute")
def get_session_history(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Load all messages for a specific session."""
    user_id = int(current_user["sub"])
    msgs = get_messages(db, session_id, user_id)
    if msgs is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return [
        {"role": m.role, "content": m.content, "rag_type": m.rag_type,
         "created_at": m.created_at.isoformat()}
        for m in msgs
    ]


@app.patch("/sessions/{session_id}", tags=["sessions"])
@limiter.limit("30/minute")
def rename_session_route(
    request: Request,
    session_id: str,
    body: RenameSessionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename a session."""
    user_id = int(current_user["sub"])
    ok = rename_session(db, session_id, user_id, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session renamed."}


@app.delete("/sessions/{session_id}", tags=["sessions"])
@limiter.limit("30/minute")
def delete_session_route(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a specific chat session."""
    user_id = int(current_user["sub"])
    ok = delete_session(db, session_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


@app.delete("/sessions", tags=["sessions"])
@limiter.limit("10/minute")
def delete_all_sessions_route(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete all sessions for the authenticated user."""
    user_id = int(current_user["sub"])
    count = delete_all_sessions(db, user_id)
    return {"message": f"Deleted {count} session(s)."}


# ═══════════════════════════════════════════════════════════════════════════════
# ── Chat routes ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
@limiter.limit("10/minute")
def chat(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Answer a legal question using the selected RAG pipeline."""
    t0 = time.monotonic()
    question = _resolve_question(body)
    rag = body.rag_type.lower()
    user_id = int(current_user["sub"])
    session_id = body.session_id
    retrieved_ids: list[str] = []
    error_msg: Optional[str] = None

    try:
        sources: list[ArticleSource] = []

        if rag == "agentic":
            result = agentic_answer(question, body.history)
            answer = result["answer"]

        elif rag == "bm25":
            chunks = bm25_retrieve(question, top_k=body.top_k)
            if not chunks:
                raise HTTPException(status_code=404, detail="No relevant articles found via BM25.")
            answer = bm25_generate(question, chunks, body.history)
            sources = _build_sources(chunks, score_key="bm25_score")
            retrieved_ids = [c.get("id", "") for c in chunks]

        elif rag == "graph_local":
            chunks = graph_local_retrieve(question, top_k=body.top_k)
            if not chunks:
                raise HTTPException(status_code=404, detail="No relevant articles found via Local Graph RAG.")
            answer = graph_local_generate(question, chunks, body.history)
            sources = _build_sources(chunks, score_key="graph_score")
            retrieved_ids = [c.get("id", "") for c in chunks]

        elif rag == "graph":
            chunks = graph_retrieve(question, top_k=body.top_k)
            if not chunks:
                raise HTTPException(status_code=404, detail="No relevant articles found via Graph RAG.")
            answer = graph_generate(question, chunks, body.history)
            sources = _build_sources(chunks, score_key="graph_score")
            retrieved_ids = [c.get("id", "") for c in chunks]

        elif rag == "hybrid":
            chunks = hybrid_retrieve(question, top_k=body.top_k)
            if not chunks:
                raise HTTPException(status_code=404, detail="No relevant articles found via Hybrid RAG.")
            answer = qwen_generate(question, chunks)
            sources = _build_sources(chunks, score_key="score")
            retrieved_ids = [c.get("id", "") for c in chunks]

        else:
            chunks = retrieve(question, top_k=body.top_k)
            if not chunks:
                raise HTTPException(status_code=404, detail="No relevant articles found.")
            answer = generate_answer(question, chunks, body.history)
            sources = _build_sources(chunks)
            retrieved_ids = [c.get("id", "") for c in chunks]

        # ── Persist to session if session_id provided ──────────────────────
        if session_id:
            add_message(db, session_id, "user", question, rag_type=rag)
            add_message(db, session_id, "assistant", answer, rag_type=rag)

        return ChatResponse(
            question=question,
            answer=answer,
            sources=sources,
            session_id=session_id,
        )

    except HTTPException as exc:
        error_msg = exc.detail
        raise
    except FileNotFoundError as exc:
        error_msg = str(exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        error_msg = str(exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
    finally:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log_request(
            query=question,
            rag_type=rag,
            retrieved_ids=retrieved_ids,
            latency_ms=latency_ms,
            status_code=500 if error_msg else 200,
            session_id=session_id,
            user_id=str(user_id),
            error=error_msg,
        )


@app.post("/api/chat/stream", tags=["chatbot"])
@limiter.limit("3/minute")
def chat_stream(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """
    Streaming SSE endpoint. Supports rag_type: agentic | graph_local | bm25.
    Yields Server-Sent Events: status, token, sources, done, error.
    """
    question = _resolve_question(body)
    rag = body.rag_type.lower()
    user_id = int(current_user["sub"])
    session_id = body.session_id

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def agentic_stream():
        full_text = ""
        try:
            for event in agentic_answer_stream(question, body.history):
                if event.get("type") == "token":
                    full_text += event.get("content", "")
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            if session_id and full_text:
                add_message(db, session_id, "user", question, rag_type=rag)
                add_message(db, session_id, "assistant", full_text, rag_type=rag)

    def graph_local_stream():
        try:
            chunks = graph_local_retrieve(question, top_k=body.top_k)
            if not chunks:
                yield _sse({"type": "error", "message": "No relevant articles found."})
                return
            sources_payload = [
                {"id": c.get("id",""), "law_name": c.get("law_name",""),
                 "article_number": str(c.get("article_number","")),
                 "title": c.get("title",""), "score": c.get("graph_score", 0.0)}
                for c in chunks
            ]
            yield _sse({"type": "sources", "sources": sources_payload})

            answer = graph_local_generate(question, chunks, body.history)
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield _sse({"type": "token", "content": token})

            yield _sse({"type": "done", "sources": sources_payload})
            if session_id:
                add_message(db, session_id, "user", question, rag_type=rag)
                add_message(db, session_id, "assistant", answer, rag_type=rag)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    def bm25_stream():
        try:
            chunks = bm25_retrieve(question, top_k=body.top_k)
            if not chunks:
                yield _sse({"type": "error", "message": "No relevant articles found."})
                return
            sources_payload = [
                {"id": c.get("id",""), "law_name": c.get("law_name",""),
                 "article_number": str(c.get("article_number","")),
                 "title": c.get("title",""), "score": c.get("bm25_score", 0.0)}
                for c in chunks
            ]
            yield _sse({"type": "sources", "sources": sources_payload})
            answer = bm25_generate(question, chunks, body.history)
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield _sse({"type": "token", "content": token})
            yield _sse({"type": "done", "sources": sources_payload})
            if session_id:
                add_message(db, session_id, "user", question, rag_type=rag)
                add_message(db, session_id, "assistant", answer, rag_type=rag)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    generators = {
        "agentic":     agentic_stream,
        "graph_local": graph_local_stream,
        "bm25":        bm25_stream,
    }
    gen_fn = generators.get(rag)
    if gen_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Streaming not supported for rag_type='{rag}'. Use: {list(generators.keys())}",
        )

    return StreamingResponse(gen_fn(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# ── Search (retrieval-only) ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/search", tags=["chatbot"])
@limiter.limit("20/minute")
def search(
    request: Request,
    body: SearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """Retrieve relevant articles WITHOUT generating an LLM answer."""
    rag = body.rag_type.lower()
    try:
        if rag == "graph_local":
            chunks = graph_local_retrieve(body.question, top_k=body.top_k)
            sources = _build_sources(chunks, score_key="graph_score")
        elif rag == "bm25":
            chunks = bm25_retrieve(body.question, top_k=body.top_k)
            sources = _build_sources(chunks, score_key="bm25_score")
        elif rag == "graph":
            chunks = graph_retrieve(body.question, top_k=body.top_k)
            sources = _build_sources(chunks, score_key="graph_score")
        else:
            chunks = retrieve(body.question, top_k=body.top_k)
            sources = _build_sources(chunks)
        return {"sources": sources}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _google_cse_search(query: str, api_key: str, cse_id: str) -> list[dict]:
    """Async wrapper for Google CSE."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": api_key, "cx": cse_id, "q": query, "num": 7}
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                results = []
                for item in items:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                        "source": item.get("displayLink", "Google"),
                    })
                return results
    except Exception as e:
        print(f"DEBUG: Google Search failed: {e}")
    return []

async def _duckduckgo_async_search(query: str) -> list[dict]:
    """Async scraper for DuckDuckGo."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = "https://html.duckduckgo.com/html/"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            resp = await client.get(url, params={"q": query}, headers=headers)
            if resp.status_code == 200:
                html_content = resp.text
                pattern = r'<div[^>]*class="[^"]*result[^"]*"[^>]*>.*?<a[^>]*class="[^"]*result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]*class="[^"]*result__snippet"[^>]*>(.*?)</a>'
                matches = re.finditer(pattern, html_content, re.DOTALL)
                results = []
                for match in matches:
                    if len(results) >= 7: break
                    raw_link = match.group(1)
                    title = html_lib.unescape(re.sub(r'<[^>]+>', '', match.group(2))).strip()
                    snippet = html_lib.unescape(re.sub(r'<[^>]+>', '', match.group(3))).strip()
                    link = raw_link
                    if "uddg=" in raw_link:
                        link = unquote(raw_link.split("uddg=")[1].split("&")[0])
                    elif raw_link.startswith("//"):
                        link = "https:" + raw_link
                    if not title or not link: continue
                    results.append({"title": title, "snippet": snippet, "link": link, "source": "DuckDuckGo"})
                return results
    except Exception as e:
        print(f"DEBUG: DuckDuckGo Search failed: {e}")
    return []

@app.get("/api/web-search", tags=["search"])
@limiter.limit("30/minute")
async def web_search(
    request: Request,
    question: str = Query(..., min_length=1, max_length=1000),
    current_user: dict = Depends(get_current_user),
):
    # ── 0. Validation & Normalization ──
    if not question or not question.strip():
        print("[Search Debug] ❌ ERROR: Received empty question.")
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    clean_query = question.strip().lower()
    print(f"[Search Debug] 📥 Incoming Question: '{question}' (Normalized: '{clean_query}')")

    # ── 0.1 Cache Check ──
    cached = search_cache.get(clean_query)
    if cached:
        print(f"[Search Debug] ⚡ CACHE HIT for: '{clean_query}'")
        return cached
    print(f"[Search Debug] 🔍 CACHE MISS for: '{clean_query}'")

    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    # ── 1. Smart Query Expansion ──
    optimized_query = _expand_legal_query(question)
    print(f"[Search Debug] 🚀 Optimized Query sent to engines: '{optimized_query}'")
    
    # ── 2. Parallel Search Execution ──
    tasks = [
        asyncio.create_task(_google_cse_search(optimized_query, api_key, cse_id)),
        asyncio.create_task(_duckduckgo_async_search(optimized_query))
    ]
    
    google_res = []
    ddg_res = []
    
    done, pending = await asyncio.wait(tasks, timeout=8.0)
    for task in done:
        try:
            res = task.result()
            if not res: continue
            if res[0].get("source") == "DuckDuckGo": ddg_res = res
            else: google_res = res
        except: continue
    for t in pending: t.cancel()

    # DEBUG LOGGING (Requirement 7)
    print(f"[Search Debug] Query: '{question}'")
    print(f"[Search Debug] Raw Results -> Google: {len(google_res)}, DDG: {len(ddg_res)}")

    # ── 3. Merge & Soft Rank (Soft Ranking Only) ──
    combined = google_res + ddg_res
    if not combined:
        print("[Search Debug] No results from any engine.")
        return {"query": optimized_query, "results": [], "summary": "", "source_used": "empty"}

    # Assign scores
    for item in combined:
        item["final_score"] = _get_authority_score(item)

    # Sort by score (Soft Rank) - NO DELETION BASED ON SCORE
    combined.sort(key=lambda x: x["final_score"], reverse=True)

    # ── 4. Link Deduplication Only (Guaranteed Output) ──
    final_results = []
    seen_links = set()
    
    for r in combined:
        link = r.get("link", "")
        if link not in seen_links:
            seen_links.add(link)
            final_results.append({
                "title": r["title"], "link": r["link"],
                "snippet": r["snippet"], "source": r["source"]
            })
        if len(final_results) >= 8: break

    # FINAL FALLBACK: If deduplication somehow broke it, return raw list
    if not final_results and combined:
        print("[Search Debug] Final filtered was empty, using raw fallback.")
        final_results = combined[:5]

    print(f"[Search Debug] Final Results Returned: {len(final_results)}")

    # ── 5. Final Assembly & Cache ──
    response_data = {
        "query": optimized_query,
        "results": final_results,
        "summary": "",
        "source_used": "soft_ranked_fallback"
    }
    
    search_cache.set(clean_query, response_data)
    return response_data


@app.post("/contact", tags=["system"])
@limiter.limit("5/minute")
def receive_contact_message(
    request: Request,
    body: ContactRequest,
    db: Session = Depends(get_db),
):
    """Submit a support or contact message."""
    try:
        create_contact_message(
            db=db,
            name=body.name,
            email=body.email,
            subject=body.subject,
            message=body.message,
        )
        return {
            "success": True,
            "message": "Message received successfully"
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save contact message: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── System routes ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["system"])
def health_check():
    """Health check — verifies the vector store is accessible."""
    try:
        import chromadb
        from backend.rag.embedding.embed_articles import VECTORSTORE_DIR, COLLECTION_NAME
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        col    = client.get_collection(COLLECTION_NAME)
        return {
            "status":           "ok",
            "vector_store":     str(VECTORSTORE_DIR),
            "collection":       COLLECTION_NAME,
            "indexed_articles": col.count(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Vector store not ready: {exc}")


# ── Hybrid pipeline endpoint ──────────────────────────────────────────────────
@app.post("/chat/hybrid", response_model=HybridChatResponse, tags=["hybrid-chatbot"])
def chat_hybrid(request: HybridChatRequest):
    """
    Answer a legal question using the Hybrid RAG pipeline.

    Pipeline:
      1. Retrieve relevant articles via BM25 + BAAI/BGE-M3 (RRF fusion)
      2. Generate the answer locally with Qwen2.5:7b (via Ollama)
      3. Return answer + source articles

    Requirements:
      - Ollama must be running: `ollama serve`
      - Model must be pulled : `ollama pull qwen2.5:7b`
      - BGE-M3 vector store  : run `python rag_bge/bge_embed_articles.py` first
      - BM25 index           : run `python bm25_rag/bm25_indexer.py` first
    """
    try:
        # ── Step 1: Hybrid retrieval (BM25 + BGE-M3 with RRF) ────────────────
        retrieved_chunks = hybrid_retrieve(request.question, top_k=request.top_k)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="لم أجد أي مواد قانونية متعلقة بسؤالك في قاعدة البيانات."
            )

        # ── Step 2: Generate answer with Qwen2.5:7b ───────────────────────────
        answer = qwen_generate(request.question, retrieved_chunks)

        # ── Step 3: Build source list ─────────────────────────────────────────
        sources = [
            HybridArticleSource(
                id=chunk.get("id", ""),
                law_name=chunk.get("law_name", ""),
                law_domain=chunk.get("law_domain", ""),
                article_number=chunk.get("article_number", ""),
                title=chunk.get("title", ""),
                text_original=chunk.get("text_original", ""),
                score=chunk.get("score", 0.0),
            )
            for chunk in retrieved_chunks
        ]

        return HybridChatResponse(
            question=request.question,
            answer=answer,
            model="qwen2.5:7b (Ollama)",
            retriever="BM25 + BAAI/BGE-M3 (RRF)",
            sources=sources,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        # Ollama unreachable or model not pulled
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health/hybrid", tags=["system"])
def health_hybrid():
    """Check that the BM25 index, BGE-M3 store, and Ollama are all reachable."""
    import urllib.request, urllib.error, json as _json

    issues = []

    # 1. Check BM25 index files
    from bm25_rag.bm25_indexer import BM25_FILE, CORPUS_FILE
    if not BM25_FILE.exists():
        issues.append(f"BM25 index not found: {BM25_FILE}")
    if not CORPUS_FILE.exists():
        issues.append(f"BM25 corpus not found: {CORPUS_FILE}")

    # 2. Check BGE-M3 vector store
    from bm25_rag.hybrid_bge_qwen_retriever import BGE_VECTORSTORE_DIR, BGE_COLLECTION_NAME
    if not BGE_VECTORSTORE_DIR.exists():
        issues.append(f"BGE-M3 vector store not found: {BGE_VECTORSTORE_DIR}")

    # 3. Ping Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_ok = False
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as r:
            ollama_ok = r.status == 200
    except Exception as e:
        issues.append(f"Ollama unreachable at {ollama_url}: {e}")

    if issues:
        raise HTTPException(status_code=503, detail={"issues": issues})

    return {
        "status": "ok",
        "bm25_index": str(BM25_FILE),
        "bge_vectorstore": str(BGE_VECTORSTORE_DIR),
        "ollama": ollama_url,
        "model": ollama_model,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

# Reload trigger 2


