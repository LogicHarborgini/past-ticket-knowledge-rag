"""
Build the vector index from the historical ticket corpus.

Run once before starting the service:

    python -m app.ingest              # upsert the corpus
    python -m app.ingest --rebuild    # drop the collection first

This is the offline half of RAG. Everything the retriever can ever find has to
be embedded here first, which makes ingestion the step where retrieval quality is
actually decided — the query side can only search what this put in.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from app.config import settings
from app.corpus import HISTORICAL_TICKETS, corpus_documents
from app.providers import active_embedding_model_id
from app.vectorstore import (
    check_manifest,
    document_count,
    get_vector_store,
    write_manifest,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Titan applies per-second request limits, and a local Ollama daemon embedding
# on CPU is simply slow. Batching keeps one throttle from failing the whole run
# and gives progress output on a corpus large enough to wait for. Fourteen
# tickets do not need it — the pattern is here because a real corpus of ten
# thousand does, and this is where that code would live.
BATCH_SIZE = 50


def batched(items: list, size: int):
    """Yield successive slices of `items`. The Day 3 batching generator, applied."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _record_corpus_in_analytics() -> None:
    """
    Mirror the corpus into the analytics database.

    This is what makes "indexed but never retrieved" answerable in SQL rather
    than by a set difference in Python. Best-effort: the vector index is the
    system of record, and a failure to write the mirror must not fail an
    ingestion that has already succeeded.
    """
    try:
        from app.analytics import sync_corpus
        from app.db import init_db, session_scope

        init_db()
        with session_scope() as session:
            count = sync_corpus(
                session,
                [ticket.to_metadata() for ticket in HISTORICAL_TICKETS],
                embedding_model=active_embedding_model_id(),
            )
        logger.info(f"Analytics corpus table synced | tickets={count}")
    except Exception as e:
        logger.warning(f"Could not sync the analytics corpus table: {e}")


def ingest(rebuild: bool = False) -> int:
    """
    Embed the corpus into the Chroma collection.

    Ingestion is an upsert: ticket IDs are used as document IDs, so re-running
    replaces each ticket in place rather than appending a second copy. Without
    stable IDs, running this twice doubles the corpus and every retrieval starts
    returning the same ticket several times over.

    `--rebuild` drops the collection first. Use it after changing the embedding
    model or the text in corpus.py — an upsert refreshes the tickets that still
    exist and silently leaves behind any that were renamed or removed.

    Returns
    -------
    int
        Number of documents in the collection afterwards.
    """
    store = get_vector_store()

    if rebuild:
        existing = document_count()
        if existing:
            logger.info(f"Dropping existing collection ({existing} documents)")
            store.reset_collection()
            store = get_vector_store()

    texts, metadatas, ids = corpus_documents()
    logger.info(
        f"Embedding {len(texts)} tickets with {active_embedding_model_id()} "
        f"into '{settings.chroma_collection}'"
    )

    start = time.perf_counter()
    for batch_no, batch in enumerate(
        zip(batched(texts, BATCH_SIZE), batched(metadatas, BATCH_SIZE), batched(ids, BATCH_SIZE)),
        start=1,
    ):
        batch_texts, batch_metadatas, batch_ids = batch
        store.add_texts(texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
        logger.info(f"  batch {batch_no}: {len(batch_texts)} tickets indexed")

    elapsed_ms = (time.perf_counter() - start) * 1000
    total = document_count()
    write_manifest(total)
    _record_corpus_in_analytics()

    logger.info(
        f"Ingestion complete | documents={total} | "
        f"elapsed={elapsed_ms:.0f}ms | persist_dir={settings.chroma_persist_dir}"
    )
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Index historical tickets into Chroma.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the collection before indexing (use after changing the embedding model)",
    )
    args = parser.parse_args()

    total = ingest(rebuild=args.rebuild)

    if total != len(HISTORICAL_TICKETS):
        # Not fatal, but worth surfacing: the usual cause is a leftover
        # collection from an earlier corpus that --rebuild would clear.
        logger.warning(
            f"Collection holds {total} documents but the corpus defines "
            f"{len(HISTORICAL_TICKETS)}. Consider `python -m app.ingest --rebuild`."
        )

    problem = check_manifest()
    if problem:
        logger.warning(problem)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
