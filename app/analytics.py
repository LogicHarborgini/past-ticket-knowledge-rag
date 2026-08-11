"""
The search log and the queries that read it.

LangSmith answers "what happened in this one run?". This answers "what has been
happening across all of them?" — cost per model this month, which corpus tickets
have never once been retrieved, whether p95 latency moved after a model change.
Traces are per-run and expire on a retention policy; this is aggregate and yours.

Three tables:

    corpus_tickets   what is in the index, written by ingestion
    searches         one row per query served
    retrievals       one row per ticket returned, with its rank

`retrievals` is a separate table rather than a comma-joined column on `searches`
because the interesting questions are per-ticket: which tickets earn their place
in the index, which never surface, which rank first most often. None of those are
answerable from a string of IDs without parsing it in application code.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────


class CorpusTicket(Base):
    """
    One indexed historical ticket, written by `python -m app.ingest`.

    Duplicating the corpus manifest into SQL is what makes "indexed but never
    retrieved" a single query rather than a Python set difference — and that
    question is the whole point of tracking retrieval. A ticket nobody ever
    matches is either dead weight in the index or, more usefully, a signal that
    the way it is written does not resemble how people describe the problem.
    """

    __tablename__ = "corpus_tickets"

    ticket_id = Column(String(32), primary_key=True)
    partner = Column(String(100))
    category = Column(String(50))
    indexed_at = Column(DateTime, nullable=False)
    embedding_model = Column(String(120), nullable=False)


class Search(Base):
    """One served query."""

    __tablename__ = "searches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # The join key back to the trace. Given a slow or wrong row here, this is what
    # turns "something went wrong on Tuesday" into the exact LangSmith run.
    langsmith_run_id = Column(String(64), nullable=True, index=True)
    ticket_id = Column(String(32), nullable=True, index=True)
    priority = Column(String(8), nullable=True)
    query = Column(Text, nullable=False)

    generation_model = Column(String(120), nullable=False, index=True)
    embedding_model = Column(String(120), nullable=False)
    top_k = Column(Integer, nullable=False)
    retrieved_count = Column(Integer, nullable=False)

    # Nullable on purpose. Not every provider reports usage — a local model
    # often does not, and the fake provider never does. A zero here would be a
    # lie that averages into every cost report; NULL is excluded from AVG and
    # SUM, which is the correct behaviour.
    prompt_tokens = Column(Integer, nullable=True)
    response_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)

    latency_ms = Column(Float, nullable=False)
    called_at = Column(DateTime, nullable=False, index=True)
    # Date bucket stored at write time as YYYY-MM-DD. Deriving it in SQL means
    # STRFTIME on SQLite and DATE_TRUNC on Postgres — one of the few places
    # SQLAlchemy does not paper over the dialect, and this project claims the
    # database is swappable. Storing the bucket keeps that claim true.
    called_on = Column(String(10), nullable=False, index=True)

    retrievals = relationship(
        "Retrieval",
        back_populates="search",
        cascade="all, delete-orphan",
    )


class Retrieval(Base):
    """One ticket returned by one search, with the position it came back in."""

    __tablename__ = "retrievals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("searches.id"), nullable=False, index=True)
    ticket_id = Column(String(32), nullable=False, index=True)
    category = Column(String(50))
    partner = Column(String(100))
    # 1-based. Rank is recorded because "retrieved" and "retrieved first" are
    # different signals: a ticket that is always third may be padding the context
    # rather than answering the question.
    rank = Column(Integer, nullable=False)

    search = relationship("Search", back_populates="retrievals")


# ─────────────────────────────────────────────────────────────────────────────
# Cost
# ─────────────────────────────────────────────────────────────────────────────

# USD per million tokens, (input, output). Published list prices at the time of
# writing — they change, and they are not the same as what an enterprise
# agreement actually bills, so treat every figure this produces as an estimate
# for relative comparison between models rather than an invoice.
#
# A model absent from this table produces cost_usd = NULL rather than a guess.
# An invented rate is worse than no rate: it looks authoritative in a report.
MODEL_RATES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic.claude-3-sonnet-20240229-v1:0": (3.00, 15.00),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.25, 1.25),
    "anthropic.claude-3-5-sonnet-20240620-v1:0": (3.00, 15.00),
    "anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00),
}


def estimate_cost_usd(
    model_id: str,
    prompt_tokens: Optional[int],
    response_tokens: Optional[int],
) -> Optional[float]:
    """
    Estimated USD cost for one call, or None when it cannot be known.

    Self-hosted models return 0.0 rather than None: there is no API charge, and
    that is a real answer to "what did this cost", not a missing one. The
    electricity is someone else's line item.
    """
    if model_id.startswith(("ollama:", "fake:")):
        return 0.0

    rates = MODEL_RATES_PER_MTOK.get(model_id)
    if rates is None or prompt_tokens is None or response_tokens is None:
        return None

    input_rate, output_rate = rates
    cost = (prompt_tokens * input_rate + response_tokens * output_rate) / 1_000_000
    return round(cost, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Writes
# ─────────────────────────────────────────────────────────────────────────────


def sync_corpus(
    session: Session,
    tickets: list[dict[str, Any]],
    embedding_model: str,
) -> int:
    """
    Record what ingestion just indexed. Upsert by ticket ID, like the vector store.

    Rows for tickets no longer in the corpus are deleted, so "never retrieved"
    cannot be answered from a ticket that was removed three ingests ago.
    """
    now = datetime.now(timezone.utc)
    current_ids = {ticket["ticket_id"] for ticket in tickets}

    for stale in session.scalars(select(CorpusTicket)).all():
        if stale.ticket_id not in current_ids:
            session.delete(stale)

    for ticket in tickets:
        row = session.get(CorpusTicket, ticket["ticket_id"])
        if row is None:
            row = CorpusTicket(ticket_id=ticket["ticket_id"])
            session.add(row)
        row.partner = ticket.get("partner")
        row.category = ticket.get("category")
        row.indexed_at = now
        row.embedding_model = embedding_model

    return len(tickets)


def log_search(
    session: Session,
    *,
    query: str,
    generation_model: str,
    embedding_model: str,
    top_k: int,
    latency_ms: float,
    sources: list[dict[str, Any]],
    ticket_id: str | None = None,
    priority: str | None = None,
    langsmith_run_id: str | None = None,
    prompt_tokens: int | None = None,
    response_tokens: int | None = None,
) -> Search:
    """
    Insert one search and its retrievals.

    The caller is responsible for treating this as best-effort — see
    `main.py`. An analytics write that fails should never cost the user an answer
    that was already generated and paid for.
    """
    now = datetime.now(timezone.utc)

    search = Search(
        langsmith_run_id=langsmith_run_id,
        ticket_id=ticket_id,
        priority=priority,
        query=query,
        generation_model=generation_model,
        embedding_model=embedding_model,
        top_k=top_k,
        retrieved_count=len(sources),
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        cost_usd=estimate_cost_usd(generation_model, prompt_tokens, response_tokens),
        latency_ms=latency_ms,
        called_at=now,
        called_on=now.strftime("%Y-%m-%d"),
    )

    for position, source in enumerate(sources, start=1):
        search.retrievals.append(
            Retrieval(
                ticket_id=source.get("ticket_id", "UNKNOWN"),
                category=source.get("category"),
                partner=source.get("partner"),
                rank=position,
            )
        )

    session.add(search)
    return search


# ─────────────────────────────────────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────────────────────────────────────


def usage_by_model(session: Session) -> list[dict[str, Any]]:
    """
    Calls, tokens, cost and latency per generation model.

        SELECT generation_model,
               COUNT(*),
               SUM(prompt_tokens), SUM(response_tokens),
               SUM(cost_usd), AVG(latency_ms), MAX(latency_ms),
               COUNT(CASE WHEN cost_usd IS NULL THEN 1 END)
        FROM searches GROUP BY generation_model ORDER BY COUNT(*) DESC

    The last column is the one that keeps the rest honest: a cost total says
    nothing without knowing how many calls could not be priced. Two unpriced
    calls out of three makes the total meaningless, and only that count reveals it.
    """
    rows = session.execute(
        select(
            Search.generation_model,
            func.count(Search.id).label("calls"),
            func.sum(Search.prompt_tokens).label("prompt_tokens"),
            func.sum(Search.response_tokens).label("response_tokens"),
            func.sum(Search.cost_usd).label("cost_usd"),
            func.avg(Search.latency_ms).label("avg_latency_ms"),
            func.max(Search.latency_ms).label("max_latency_ms"),
            # CASE WHEN inside COUNT — counts only the rows matching the
            # condition, because COUNT ignores NULL.
            func.count(Search.id).filter(Search.cost_usd.is_(None)).label("unpriced_calls"),
        )
        .group_by(Search.generation_model)
        .order_by(func.count(Search.id).desc())
    ).all()

    return [
        {
            "generation_model": row.generation_model,
            "calls": row.calls,
            "prompt_tokens": row.prompt_tokens,
            "response_tokens": row.response_tokens,
            "cost_usd": round(row.cost_usd, 6) if row.cost_usd is not None else None,
            "avg_latency_ms": round(row.avg_latency_ms, 1),
            "max_latency_ms": round(row.max_latency_ms, 1),
            "unpriced_calls": row.unpriced_calls,
        }
        for row in rows
    ]


def latency_percentile_by_model(
    session: Session, percentile: float = 0.95
) -> list[dict[str, Any]]:
    """
    Latency at a percentile, per model — the metric an SLO is written against.

    SQLite has no PERCENTILE function, so this is done with window functions:

        WITH ranked AS (
            SELECT generation_model, latency_ms,
                   ROW_NUMBER() OVER (PARTITION BY generation_model
                                      ORDER BY latency_ms) AS rn,
                   COUNT(*)     OVER (PARTITION BY generation_model) AS n
            FROM searches
        )
        SELECT generation_model, MIN(latency_ms)
        FROM ranked
        WHERE rn >= n * 0.95
        GROUP BY generation_model

    Rank every row within its model, count the rows in the same window without
    collapsing them, then take the first row at or past the cut. This is the
    thing window functions do that GROUP BY cannot: per-row context about the
    group while keeping the rows.

    Averages hide exactly what this exposes. A 900ms mean with a 6s p95 is a
    service that feels broken to one user in twenty, and the mean will never say so.
    """
    ranked = (
        select(
            Search.generation_model.label("model"),
            Search.latency_ms.label("latency_ms"),
            func.row_number()
            .over(partition_by=Search.generation_model, order_by=Search.latency_ms)
            .label("rn"),
            func.count()
            .over(partition_by=Search.generation_model)
            .label("n"),
        )
    ).subquery()

    rows = session.execute(
        select(
            ranked.c.model,
            func.min(ranked.c.latency_ms).label("latency_ms"),
            func.max(ranked.c.n).label("samples"),
        )
        .where(ranked.c.rn >= ranked.c.n * percentile)
        .group_by(ranked.c.model)
    ).all()

    return [
        {
            "generation_model": row.model,
            "percentile": percentile,
            "latency_ms": round(row.latency_ms, 1),
            "samples": row.samples,
        }
        for row in rows
    ]


def top_retrieved_tickets(session: Session, limit: int = 10) -> list[dict[str, Any]]:
    """
    Which corpus tickets are retrieved most, and how often they rank first.

        SELECT ticket_id, COUNT(*), SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END)
        FROM retrievals GROUP BY ticket_id ORDER BY COUNT(*) DESC
    """
    rows = session.execute(
        select(
            Retrieval.ticket_id,
            Retrieval.category,
            func.count(Retrieval.id).label("times_retrieved"),
            func.count(Retrieval.id).filter(Retrieval.rank == 1).label("times_first"),
            func.avg(Retrieval.rank).label("avg_rank"),
        )
        .group_by(Retrieval.ticket_id, Retrieval.category)
        .order_by(func.count(Retrieval.id).desc())
        .limit(limit)
    ).all()

    return [
        {
            "ticket_id": row.ticket_id,
            "category": row.category,
            "times_retrieved": row.times_retrieved,
            "times_ranked_first": row.times_first,
            "avg_rank": round(row.avg_rank, 2),
        }
        for row in rows
    ]


def never_retrieved_tickets(session: Session) -> list[dict[str, Any]]:
    """
    Indexed tickets that no search has ever returned.

        SELECT c.ticket_id, c.category, c.partner
        FROM corpus_tickets c
        LEFT JOIN retrievals r ON c.ticket_id = r.ticket_id
        WHERE r.ticket_id IS NULL

    LEFT JOIN + IS NULL rather than NOT IN: `NOT IN` against a subquery
    containing a single NULL returns no rows at all, silently, because
    `x NOT IN (1, NULL)` evaluates to NULL rather than true. This form has no
    such failure mode.

    For a RAG system this is the most useful query in the file. A ticket that
    never surfaces is either genuinely irrelevant to what people ask, or written
    in language nobody uses — and the second is a fixable retrieval problem
    hiding as a content problem.
    """
    rows = session.execute(
        select(
            CorpusTicket.ticket_id,
            CorpusTicket.category,
            CorpusTicket.partner,
            CorpusTicket.indexed_at,
        )
        .outerjoin(Retrieval, CorpusTicket.ticket_id == Retrieval.ticket_id)
        .where(Retrieval.ticket_id.is_(None))
        .order_by(CorpusTicket.ticket_id)
    ).all()

    return [
        {
            "ticket_id": row.ticket_id,
            "category": row.category,
            "partner": row.partner,
            "indexed_at": row.indexed_at.isoformat() if row.indexed_at else None,
        }
        for row in rows
    ]


def daily_volume_with_growth(session: Session) -> list[dict[str, Any]]:
    """
    Searches per day, with the day-over-day change.

        SELECT called_on, COUNT(*) AS searches,
               LAG(COUNT(*)) OVER (ORDER BY called_on) AS previous_day
        FROM searches GROUP BY called_on ORDER BY called_on

    LAG reads the previous row of the result set, which is how a growth column is
    written without a self-join. Note it operates on the *aggregated* rows —
    window functions are evaluated after GROUP BY, which is why COUNT(*) can be
    nested inside LAG here and could not be in a WHERE clause.
    """
    daily = (
        select(
            Search.called_on.label("day"),
            func.count(Search.id).label("searches"),
        )
        .group_by(Search.called_on)
        .subquery()
    )

    rows = session.execute(
        select(
            daily.c.day,
            daily.c.searches,
            func.lag(daily.c.searches).over(order_by=daily.c.day).label("previous_day"),
        ).order_by(daily.c.day)
    ).all()

    results = []
    for row in rows:
        previous = row.previous_day
        # Guard the division: a growth percentage against a zero baseline is
        # infinity, not a number worth putting in a report.
        change_pct = (
            round(100.0 * (row.searches - previous) / previous, 1)
            if previous else None
        )
        results.append({
            "day": row.day,
            "searches": row.searches,
            "previous_day": previous,
            "change_pct": change_pct,
        })
    return results


def slowest_searches_per_model(session: Session, n: int = 3) -> list[dict[str, Any]]:
    """
    The n slowest searches for each model — top-N per group.

        WITH ranked AS (
            SELECT id, generation_model, query, latency_ms,
                   ROW_NUMBER() OVER (PARTITION BY generation_model
                                      ORDER BY latency_ms DESC) AS rn
            FROM searches
        )
        SELECT * FROM ranked WHERE rn <= :n

    ROW_NUMBER rather than DENSE_RANK: latencies are floats and ties are
    vanishingly unlikely, so the tie-handling DENSE_RANK buys is not worth
    returning an unbounded number of rows for. On a column where ties are
    meaningful — cost per call, say — DENSE_RANK would be the right choice.

    The `langsmith_run_id` on each row is the point: this query finds the slow
    requests, and that ID opens the trace that explains them.
    """
    ranked = (
        select(
            Search.id,
            Search.generation_model,
            Search.query,
            Search.latency_ms,
            Search.retrieved_count,
            Search.langsmith_run_id,
            Search.called_at,
            func.row_number()
            .over(
                partition_by=Search.generation_model,
                order_by=Search.latency_ms.desc(),
            )
            .label("rn"),
        )
    ).subquery()

    rows = session.execute(
        select(ranked).where(ranked.c.rn <= n).order_by(
            ranked.c.generation_model, ranked.c.rn
        )
    ).all()

    return [
        {
            "generation_model": row.generation_model,
            "query": row.query[:120],
            "latency_ms": round(row.latency_ms, 1),
            "retrieved_count": row.retrieved_count,
            "langsmith_run_id": row.langsmith_run_id,
            "called_at": row.called_at.isoformat() if row.called_at else None,
        }
        for row in rows
    ]
