"""
Chroma vector store and the retriever built on it.

Chroma stands in for Amazon OpenSearch k-NN. The interface LangChain exposes is
the same on both — `as_retriever()` returns a Runnable that takes a query string
and returns Documents — so the chain in ptk_chain.py never learns which one it
is talking to. Swapping in OpenSearchVectorSearch is a change to this file and
nothing else. That separation is the reason the Retriever abstraction exists.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from app.config import settings
from app.embeddings import build_embeddings
from app.providers import active_embedding_model_id

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "ptk_index_manifest.json"


def persist_dir() -> Path:
    return Path(settings.chroma_persist_dir)


def manifest_path() -> Path:
    return persist_dir() / MANIFEST_FILENAME


def write_manifest(document_count: int) -> None:
    """
    Record which embedding model built this index.

    Vectors carry no record of the model that produced them, so an index built
    with Titan and queried with nomic-embed-text returns confident nonsense: the
    similarity maths is perfectly valid over two unrelated coordinate systems.
    Writing the model down is what turns that into a warning at startup instead
    of a bug report about "irrelevant results".
    """
    persist_dir().mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(
        json.dumps(
            {
                "embedding_model": active_embedding_model_id(),
                "collection": settings.chroma_collection,
                "document_count": document_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def check_manifest() -> str | None:
    """
    Compare the current embedding model against the one the index was built with.

    Returns
    -------
    str | None
        A human-readable problem description, or None when the index is absent
        (nothing to disagree with yet) or consistent.
    """
    path = manifest_path()
    if not path.exists():
        return None

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"index manifest unreadable ({e}) — rebuild with `python -m app.ingest --rebuild`"

    indexed_with = manifest.get("embedding_model")
    current = active_embedding_model_id()
    if indexed_with and indexed_with != current:
        return (
            f"index was built with '{indexed_with}' but the active embedding model "
            f"is '{current}' — retrieval results will be meaningless. "
            "Rebuild with `python -m app.ingest --rebuild`."
        )

    return None


def get_vector_store() -> VectorStore:
    """
    Open (or create) the Chroma collection.

    Not cached. Chroma's client holds a handle on the persisted directory, and a
    module-level singleton makes the store built during ingestion outlive the
    ingest process in tests — each caller getting its own handle costs a few
    milliseconds and removes a class of cross-test bleed.
    """
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=build_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        # Chroma indexes with squared L2 by default. Cosine is what the embedding
        # models are trained for — it compares direction and ignores magnitude, so
        # a long ticket and a short query are judged on meaning rather than length.
        collection_metadata={"hnsw:space": "cosine"},
    )


def document_count() -> int:
    """How many documents are currently indexed. 0 when ingestion has not run."""
    try:
        store = get_vector_store()
        return store._collection.count()   # noqa: SLF001 - no public count() on the wrapper
    except Exception as e:
        logger.warning(f"Could not read collection count: {e}")
        return 0


def get_retriever(k: int | None = None) -> Runnable:
    """
    The retriever the chain consumes.

    search_type is plain similarity. MMR is the alternative — it trades some
    similarity for diversity between the results — and it earns its keep on a
    corpus with many near-duplicate tickets, which this one deliberately is not.
    On fourteen distinct tickets MMR would push a less relevant result into the
    top-k purely for the sake of variety.

    With RETRIEVAL_SCORE_THRESHOLD set, the retriever drops matches below the
    cutoff and can legitimately return nothing. That is the point: a plain
    similarity search always returns its k nearest neighbours no matter how far
    away they are, so a question the corpus has no answer to still comes back
    with three confident-looking tickets about something else. The threshold is
    how "we have nothing on this" becomes representable.
    """
    k = k or settings.retrieval_top_k
    threshold = settings.retrieval_score_threshold

    if threshold is None:
        return get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    return get_vector_store().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": threshold},
    )
