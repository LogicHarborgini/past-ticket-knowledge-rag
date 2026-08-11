"""
Database engine and session management for the analytics log.

SQLite by default, because the analytics store is a local artefact you can open
in DB Browser and query by hand. `PTK_DATABASE_URL` points it at Postgres or
anything else SQLAlchemy speaks — no code here knows which, which is the reason
for using SQLAlchemy at all rather than the `sqlite3` module.

Two ways to get a session, deliberately:

    session_scope()   a context manager, for scripts and background work
    get_db_session()  a FastAPI dependency, one session per request

Both close the session in a `finally`. A leaked session holds a connection open,
and under SQLite that means holding a write lock — the failure shows up much
later as an unrelated request timing out.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


def _engine_kwargs(url: str) -> dict:
    """
    SQLite needs two arguments no other database does.

    check_same_thread=False: SQLite forbids using a connection from a thread
    other than the one that opened it. FastAPI runs sync dependency code in a
    thread pool, so the connection legitimately moves between threads and the
    default guard rejects it. SQLAlchemy's pool keeps one session on one thread
    at a time, which is the property the guard exists to protect.
    """
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


engine = create_engine(
    settings.database_url,
    # echo=False: SQL logging is genuinely useful while developing a query and
    # unbearable in a request log. Flip it per-session in a REPL instead.
    echo=False,
    future=True,
    **_engine_kwargs(settings.database_url),
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """
    Create the analytics tables if they do not exist.

    create_all is enough for a single append-only log with no schema evolution
    to manage. The moment a column needs to change shape, this becomes Alembic —
    create_all will not alter an existing table, and will silently do nothing
    while the code expects a column that is not there.
    """
    from app.analytics import Base   # imported here to avoid a circular import

    Base.metadata.create_all(bind=engine)
    logger.info("Analytics tables ready | url=%s", settings.database_url)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Session that commits on success, rolls back on error, always closes."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency: one session per request, closed when the response is sent.

    No commit here — the analytics endpoints only read. A dependency that commits
    on the way out would turn a failed read into a write of whatever happened to
    be pending in the session.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
