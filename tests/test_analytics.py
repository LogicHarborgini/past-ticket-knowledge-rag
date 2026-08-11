"""
Tests for the search log and the queries over it.

These run against a real SQLite database, not a mocked session. The whole value
of this module is in the SQL — window functions, an outer join, NULL handling —
and a mocked session tests none of it.
"""

from datetime import datetime, timezone

import pytest

from app.analytics import (
    CorpusTicket,
    Search,
    daily_volume_with_growth,
    estimate_cost_usd,
    latency_percentile_by_model,
    log_search,
    never_retrieved_tickets,
    slowest_searches_per_model,
    top_retrieved_tickets,
    usage_by_model,
)


def _sources(*ticket_ids: str) -> list[dict]:
    return [
        {"ticket_id": tid, "category": "Database", "partner": "Acme Corp", "excerpt": "..."}
        for tid in ticket_ids
    ]


def _log(session, **overrides):
    payload = {
        "query": "connection pool exhausted",
        "generation_model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "top_k": 3,
        "latency_ms": 1000.0,
        "sources": _sources("HIST-009"),
    }
    payload.update(overrides)
    search = log_search(session, **payload)
    session.commit()
    return search


# ─────────────────────────────────────────────────────────────────────────────
# Cost
# ─────────────────────────────────────────────────────────────────────────────


def test_cost_is_computed_from_published_rates():
    # Sonnet: $3/MTok input, $15/MTok output.
    cost = estimate_cost_usd("anthropic.claude-3-sonnet-20240229-v1:0", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


def test_unknown_model_has_no_cost_rather_than_a_guessed_one():
    """
    A fabricated rate is worse than a missing one: it looks authoritative in a
    report and nothing downstream can tell it was invented.
    """
    assert estimate_cost_usd("some-model-nobody-priced", 1000, 500) is None


def test_missing_token_counts_produce_no_cost():
    assert estimate_cost_usd("anthropic.claude-3-sonnet-20240229-v1:0", None, None) is None


def test_self_hosted_models_cost_zero_not_unknown():
    """0.0 is a real answer for a local model; None would mean 'we don't know'."""
    assert estimate_cost_usd("ollama:llama3.2", 1000, 500) == 0.0
    assert estimate_cost_usd("fake:canned-answer", None, None) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Writes
# ─────────────────────────────────────────────────────────────────────────────


def test_search_and_its_retrievals_are_persisted(db_session):
    search = _log(db_session, sources=_sources("HIST-009", "HIST-010", "HIST-003"))

    assert search.id is not None
    assert search.retrieved_count == 3
    assert [r.ticket_id for r in search.retrievals] == ["HIST-009", "HIST-010", "HIST-003"]
    # Rank is 1-based and follows retrieval order — "retrieved" and "retrieved
    # first" are different signals.
    assert [r.rank for r in search.retrievals] == [1, 2, 3]


def test_run_id_is_stored_so_a_row_can_be_traced_back(db_session):
    search = _log(db_session, langsmith_run_id="abc-123")
    assert search.langsmith_run_id == "abc-123"


def test_unknown_token_counts_are_stored_as_null(db_session):
    """
    Zero would average into token and cost reports as a measurement. NULL is
    skipped by SUM and AVG, which is the behaviour that keeps totals honest.
    """
    search = _log(db_session, prompt_tokens=None, response_tokens=None)
    assert search.prompt_tokens is None
    assert search.cost_usd is None


# ─────────────────────────────────────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────────────────────────────────────


def test_usage_by_model_reports_unpriced_calls(db_session):
    _log(db_session, prompt_tokens=1000, response_tokens=500)
    _log(db_session, prompt_tokens=None, response_tokens=None)

    rows = usage_by_model(db_session)
    assert len(rows) == 1

    row = rows[0]
    assert row["calls"] == 2
    assert row["unpriced_calls"] == 1
    # SUM skipped the NULL row rather than treating it as zero.
    assert row["prompt_tokens"] == 1000


def test_latency_percentile_picks_the_row_at_the_cut(db_session):
    """
    Twenty samples at 100..2000ms. p95 is the 19th value, 1900 — not the mean,
    which would be 1050 and would describe an experience nobody had.
    """
    for i in range(1, 21):
        _log(db_session, latency_ms=float(i * 100))

    rows = latency_percentile_by_model(db_session, percentile=0.95)
    assert len(rows) == 1
    assert rows[0]["samples"] == 20
    assert rows[0]["latency_ms"] == 1900.0


def test_latency_percentile_partitions_by_model(db_session):
    for i in range(1, 11):
        _log(db_session, latency_ms=float(i * 10), generation_model="ollama:llama3.2")
    for i in range(1, 11):
        _log(db_session, latency_ms=float(i * 1000))

    rows = {row["generation_model"]: row["latency_ms"] for row in latency_percentile_by_model(db_session)}
    assert rows["ollama:llama3.2"] == 100.0
    assert rows["anthropic.claude-3-sonnet-20240229-v1:0"] == 10000.0


def test_top_retrieved_tickets_counts_first_place_separately(db_session):
    _log(db_session, sources=_sources("HIST-009", "HIST-010"))
    _log(db_session, sources=_sources("HIST-010", "HIST-009"))
    _log(db_session, sources=_sources("HIST-009"))

    rows = {row["ticket_id"]: row for row in top_retrieved_tickets(db_session)}
    assert rows["HIST-009"]["times_retrieved"] == 3
    assert rows["HIST-009"]["times_ranked_first"] == 2
    assert rows["HIST-010"]["times_ranked_first"] == 1


def test_never_retrieved_finds_indexed_tickets_no_search_returned(db_session):
    """
    The LEFT JOIN + IS NULL pattern. NOT IN would be the obvious alternative and
    returns nothing at all if the subquery contains a single NULL.
    """
    indexed = db_session.query(CorpusTicket).count()
    assert indexed > 0, "ingestion should have populated corpus_tickets"

    _log(db_session, sources=_sources("HIST-009"))

    missing = {row["ticket_id"] for row in never_retrieved_tickets(db_session)}
    assert "HIST-009" not in missing
    assert len(missing) == indexed - 1


def test_daily_volume_reports_change_against_the_previous_day(db_session):
    """LAG reads the previous aggregated row — no self-join needed."""
    for day, count in (("2026-08-01", 2), ("2026-08-02", 3)):
        for _ in range(count):
            search = _log(db_session)
            search.called_on = day
            search.called_at = datetime(2026, 8, int(day[-2:]), tzinfo=timezone.utc)
    db_session.commit()

    rows = daily_volume_with_growth(db_session)
    assert [row["day"] for row in rows] == ["2026-08-01", "2026-08-02"]
    assert rows[0]["previous_day"] is None      # nothing before the first day
    assert rows[0]["change_pct"] is None
    assert rows[1]["previous_day"] == 2
    assert rows[1]["change_pct"] == 50.0


def test_slowest_searches_returns_n_per_model(db_session):
    for i in range(1, 6):
        _log(db_session, latency_ms=float(i * 100))
    for i in range(1, 6):
        _log(db_session, latency_ms=float(i * 10), generation_model="ollama:llama3.2")

    rows = slowest_searches_per_model(db_session, n=2)
    assert len(rows) == 4   # 2 per model, 2 models

    by_model: dict[str, list[float]] = {}
    for row in rows:
        by_model.setdefault(row["generation_model"], []).append(row["latency_ms"])

    # Descending within each partition, capped at n.
    assert by_model["ollama:llama3.2"] == [50.0, 40.0]
    assert by_model["anthropic.claude-3-sonnet-20240229-v1:0"] == [500.0, 400.0]


def test_reads_return_empty_rather_than_failing_on_an_empty_log(db_session):
    """A fresh deployment has no rows, and a dashboard should render regardless."""
    assert usage_by_model(db_session) == []
    assert latency_percentile_by_model(db_session) == []
    assert daily_volume_with_growth(db_session) == []
    assert slowest_searches_per_model(db_session) == []
    assert db_session.query(Search).count() == 0
