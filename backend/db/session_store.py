"""
session_store.py
----------------
CRUD helpers for users and chat sessions, consumed by API routes.

All DB operations are synchronous (SQLite + SQLAlchemy Core Sessions).
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import bcrypt
from sqlalchemy.orm import Session

from backend.db.models import ChatMessage, ChatSession, User, ContactMessage

# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(db: Session, username: str, email: str, password: str) -> User:
    user = User(
        username=username,
        email=email.lower().strip(),
        hashed_pw=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower().strip()).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_pw):
        return None
    return user


def delete_user(db: Session, user_id: int) -> bool:
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True


def update_password(db: Session, user_id: int, new_password: str) -> bool:
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    user.hashed_pw = hash_password(new_password)
    db.commit()
    return True


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(db: Session, user_id: int, title: str = "New Conversation") -> ChatSession:
    session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_sessions(db: Session, user_id: int, limit: int = 50) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )


def get_session(db: Session, session_id: str, user_id: int) -> Optional[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )


def rename_session(db: Session, session_id: str, user_id: int, new_title: str) -> bool:
    sess = get_session(db, session_id, user_id)
    if not sess:
        return False
    sess.title = new_title[:128]
    db.commit()
    return True


def delete_session(db: Session, session_id: str, user_id: int) -> bool:
    sess = get_session(db, session_id, user_id)
    if not sess:
        return False
    db.delete(sess)
    db.commit()
    return True


def delete_all_sessions(db: Session, user_id: int) -> int:
    """Delete all sessions for a user. Returns the number deleted."""
    count = db.query(ChatSession).filter(ChatSession.user_id == user_id).count()
    db.query(ChatSession).filter(ChatSession.user_id == user_id).delete()
    db.commit()
    return count


# ── Message CRUD ──────────────────────────────────────────────────────────────

def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    rag_type: str = "graph_local",
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        rag_type=rag_type,
    )
    db.add(msg)

    # Bump session.updated_at and auto-title from first user message
    sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if sess:
        sess.updated_at = datetime.now(timezone.utc)
        if role == "user" and sess.title == "New Conversation":
            sess.title = content[:60]

    db.commit()
    db.refresh(msg)
    return msg


def get_messages(db: Session, session_id: str, user_id: int) -> list[ChatMessage]:
    """Returns all messages in a session, verifying ownership."""
    sess = get_session(db, session_id, user_id)
    if not sess:
        return []
    return sess.messages


# ── Contact Message CRUD ──────────────────────────────────────────────────────

def create_contact_message(
    db: Session,
    name: str,
    email: str,
    subject: Optional[str],
    message: str,
) -> ContactMessage:
    msg = ContactMessage(
        name=name,
        email=email.lower().strip(),
        subject=subject,
        message=message,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


# ── Google Authentication CRUD ───────────────────────────────────────────────

def get_user_by_google_id(db: Session, google_id: str) -> Optional[User]:
    """Retrieve a user by their Google ID."""
    return db.query(User).filter(User.google_id == google_id).first()


def create_google_user(db: Session, username: str, email: str, google_id: str) -> User:
    """Create a new user registered via Google with a secure random password."""
    user = User(
        username=username.strip(),
        email=email.lower().strip(),
        hashed_pw=hash_password(str(uuid.uuid4())),
        google_id=google_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def link_google_account(db: Session, user: User, google_id: str) -> User:
    """Link an existing user account to a Google ID."""
    user.google_id = google_id
    db.commit()
    db.refresh(user)
    return user
