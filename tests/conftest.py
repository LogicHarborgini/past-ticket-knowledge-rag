"""
Shared test configuration.

pytest imports conftest.py before any test module, which is the only window in
which these can be set usefully: app.config calls load_dotenv() at import time,
and load_dotenv does not overwrite variables already present in os.environ. So
setting them here wins over .env, and pydantic-settings also reads os.environ
ahead of the .env file.

The point is that the suite must run offline: no LangSmith network calls, no AWS
credentials, no local model, and no dependency on whatever happens to be in the
developer's ./chroma_db. A test suite that needs any of those is a test suite
that fails on someone else's machine.
"""

import os
import tempfile
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LLM_PROVIDER"] = "fake"

# A per-session Chroma directory, created before app.config is imported. Pointing
# the tests at the real ./chroma_db would make results depend on when ingestion
# last ran, and — worse — let a test's writes land in the developer's index.
_TEST_INDEX_DIR = tempfile.mkdtemp(prefix="ptk-test-chroma-")
os.environ["CHROMA_PERSIST_DIR"] = _TEST_INDEX_DIR
os.environ["CHROMA_COLLECTION"] = "past_tickets_test"

# Same reasoning for the analytics log: tests must not write rows into the
# developer's ./ptk_analytics.db, and the coverage assertions need a database
# whose contents they control rather than one holding yesterday's searches.
_TEST_DB_PATH = Path(tempfile.mkdtemp(prefix="ptk-test-db-")) / "analytics.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH.as_posix()}"

import pytest  # noqa: E402 - must follow the env setup above

from app.corpus import HISTORICAL_TICKETS  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.ingest import ingest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def indexed_corpus() -> int:
    """
    Ingest the corpus once for the whole session.

    Session-scoped because embedding the corpus, even with keyword embeddings, is
    the most expensive thing the suite does — and every retrieval test wants the
    same index. autouse because a test that silently ran against an empty index
    would pass its assertions about "no results" for entirely the wrong reason.
    """
    count = ingest(rebuild=True)
    assert count == len(HISTORICAL_TICKETS)
    return count


@pytest.fixture(scope="session")
def test_index_dir() -> Path:
    return Path(_TEST_INDEX_DIR)


@pytest.fixture
def db_session():
    """
    A session against the test analytics database, cleaned before each test.

    Truncated rather than recreated per test: the tables are created once by
    ingestion, and deleting rows is both faster and closer to how the log behaves
    in life — an append-only table that has already existed for a while.

    corpus_tickets is deliberately left alone. Ingestion populates it, and the
    coverage tests need it there.
    """
    from app.analytics import Retrieval, Search

    init_db()
    session = SessionLocal()
    try:
        session.query(Retrieval).delete()
        session.query(Search).delete()
        session.commit()
        yield session
    finally:
        session.close()
