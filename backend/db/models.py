"""
models.py
---------
SQLAlchemy ORM models — Users and Chat Sessions.

Database: SQLite (fully offline, zero external dependencies).
File:     backend/db/istacherni.db
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

# ── Database path & engine ────────────────────────────────────────────────────
DB_DIR  = Path(__file__).parent
DB_FILE = DB_DIR / "istacherni.db"
DB_URL  = f"sqlite:///{DB_FILE}"

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + threading
    echo=False,
)


class Base(DeclarativeBase):
    pass


# ── User ──────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64), unique=True, index=True, nullable=False)
    email      = Column(String(256), unique=True, index=True, nullable=False)
    hashed_pw  = Column(String(256), nullable=False)
    is_active  = Column(Boolean, default=True)
    google_id  = Column(String(256), unique=True, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    sessions = relationship("ChatSession", back_populates="owner", cascade="all, delete-orphan")


# ── Chat Session ──────────────────────────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id         = Column(String(64), primary_key=True, index=True)   # UUID string
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title      = Column(String(128), default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    owner    = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")


# ── Chat Message ──────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role       = Column(String(16), nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    rag_type   = Column(String(32), default="graph_local")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


# ── Contact Message ───────────────────────────────────────────────────────────

class ContactMessage(Base):
    __tablename__ = "contact_messages"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(128), nullable=False)
    email      = Column(String(256), nullable=False)
    subject    = Column(String(256), nullable=True)
    message    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── Create all tables on import ───────────────────────────────────────────────

def init_db() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    Base.metadata.create_all(bind=engine)
    
    # Safe SQLite migration to add google_id column if it doesn't exist
    from sqlalchemy import text
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR(256)"))
            conn.execute(text("CREATE UNIQUE INDEX ix_users_google_id ON users (google_id)"))
            print("[Database] Safe Migration: Added google_id column to users table.")
        except Exception:
            # Column already exists or another error occurs (safely ignore)
            pass


def get_db() -> Session:
    """FastAPI dependency: yields a SQLAlchemy Session and closes it after the request."""
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()
