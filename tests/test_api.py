"""
Tests for the FastAPI layer.

TestClient drives the real app including its lifespan, so these cover the two
things unit tests on the chain cannot: that Pydantic rejects bad input before any
embedding call is made, and that the response contract holds.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    # `with` runs the lifespan handler, which is where the index checks live.
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_populated_index(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["service"] == "PTK"
    assert body["indexed_documents"] > 0
    assert body["index_warning"] is None
    # "healthy" here means more than "the process is up": a RAG service with an
    # empty index answers every query with nothing and passes a liveness check.
    assert body["status"] == "healthy"


def test_search_returns_answer_and_sources(client: TestClient):
    response = client.post(
        "/api/v1/search",
        json={
            "query": "Connection pool exhausted on production, batch job cannot start",
            "ticket_id": "tick-9001",
            "priority": "P1",
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["answer"]
    assert body["retrieved_count"] == len(body["sources"])
    assert body["sources"], "an answer with no sources is an ungrounded answer"
    assert body["model_used"] == "fake:canned-answer"
    assert body["embedding_model"] == "fake:keyword-embeddings"
    assert body["latency_ms"] > 0
    # Tracing is off in the suite, so there is no trace to reference.
    assert body["langsmith_run_id"] is None


def test_search_cites_the_expected_ticket(client: TestClient):
    response = client.post(
        "/api/v1/search",
        json={"query": "AS2 messages send but no MDN receipts come back from the partner"},
    )
    cited = [source["ticket_id"] for source in response.json()["sources"]]
    assert "HIST-003" in cited


def test_short_query_is_rejected_before_the_model(client: TestClient):
    """422, not 503 — a bad request should never reach the embedding call."""
    response = client.post("/api/v1/search", json={"query": "db down"})
    assert response.status_code == 422


def test_invalid_priority_is_rejected(client: TestClient):
    response = client.post(
        "/api/v1/search",
        json={
            "query": "Connection pool exhausted on production servers",
            "priority": "URGENT",
        },
    )
    assert response.status_code == 422


def test_top_k_above_the_cap_is_rejected(client: TestClient):
    response = client.post(
        "/api/v1/search",
        json={"query": "Connection pool exhausted on production servers", "top_k": 99},
    )
    assert response.status_code == 422


def test_top_k_is_honoured(client: TestClient):
    response = client.post(
        "/api/v1/search",
        json={"query": "Connection pool exhausted on production servers", "top_k": 1},
    )
    assert response.json()["retrieved_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Analytics endpoints
#
# These run after the search tests above and read what those searches logged, so
# they assert on shape and on the invariants that must hold for any traffic —
# not on exact counts, which would couple them to the order tests happen to run in.
# ─────────────────────────────────────────────────────────────────────────────


def test_search_is_written_to_the_analytics_log(client: TestClient):
    client.post(
        "/api/v1/search",
        json={
            "query": "AS2 MDN receipts are missing entirely from the partner",
            "ticket_id": "tick-log-1",
            "priority": "P2",
        },
    )

    rows = client.get("/api/v1/analytics/usage").json()
    assert rows, "the search above should have produced a usage row"
    assert rows[0]["generation_model"] == "fake:canned-answer"
    assert rows[0]["calls"] >= 1


def test_usage_endpoint_reports_unpriced_calls(client: TestClient):
    row = client.get("/api/v1/analytics/usage").json()[0]
    # The fake provider reports no token usage, so tokens are NULL — but its
    # cost is a known 0.0 rather than unknown, so nothing is unpriced.
    assert row["prompt_tokens"] is None
    assert row["unpriced_calls"] == 0


def test_latency_percentile_endpoint(client: TestClient):
    rows = client.get("/api/v1/analytics/latency?percentile=0.9").json()
    assert rows
    assert rows[0]["percentile"] == 0.9
    assert rows[0]["latency_ms"] > 0


def test_latency_percentile_rejects_an_out_of_range_percentile(client: TestClient):
    assert client.get("/api/v1/analytics/latency?percentile=1.5").status_code == 422


def test_retrieval_coverage_reports_indexed_and_unretrieved(client: TestClient):
    body = client.get("/api/v1/analytics/retrieval-coverage").json()

    assert body["indexed_tickets"] == 14
    assert body["retrieved_at_least_once"] <= body["indexed_tickets"]
    assert (
        body["retrieved_at_least_once"] + len(body["never_retrieved"])
        == body["indexed_tickets"]
    ), "every indexed ticket is either retrieved at some point or never retrieved"
    assert 0.0 <= body["coverage_pct"] <= 100.0


def test_volume_endpoint_returns_a_day_bucket(client: TestClient):
    rows = client.get("/api/v1/analytics/volume").json()
    assert rows
    assert len(rows[0]["day"]) == 10          # YYYY-MM-DD
    assert rows[0]["previous_day"] is None    # nothing precedes the first day


def test_slowest_endpoint_caps_rows_per_model(client: TestClient):
    rows = client.get("/api/v1/analytics/slowest?n=2").json()
    per_model: dict[str, int] = {}
    for row in rows:
        per_model[row["generation_model"]] = per_model.get(row["generation_model"], 0) + 1
    assert all(count <= 2 for count in per_model.values())


def test_analytics_failure_never_breaks_a_search(client: TestClient, monkeypatch):
    """
    Logging is a side channel. The answer has already been generated and paid
    for by the time the insert runs, so a database problem must cost an analytics
    row and nothing else.
    """
    import app.main as main

    def explode(*args, **kwargs):
        raise RuntimeError("analytics database is on fire")

    monkeypatch.setattr(main.analytics, "log_search", explode)

    response = client.post(
        "/api/v1/search",
        json={"query": "Connection pool exhausted on production servers"},
    )
    assert response.status_code == 200
    assert response.json()["answer"]
