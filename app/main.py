"""
FastAPI application for Past Ticket Knowledge.

A RAG service over resolved historical support tickets.
Stack: FastAPI + LangChain LCEL + Chroma (OpenSearch in production) + Bedrock.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import analytics
from app.config import settings
from app.db import get_db_session, init_db, session_scope
from app.models import (
    DailyVolumeRow,
    HealthResponse,
    LatencyPercentileRow,
    ModelUsageRow,
    PTKSearchRequest,
    PTKSearchResponse,
    RetrievalCoverageResponse,
    RetrievedTicket,
    SlowSearchRow,
)
from app.providers import active_embedding_model_id, active_model_id
from app.ptk_chain import ainvoke_ptk_traced
from app.vectorstore import check_manifest, document_count

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup checks.

    The LangSmith check reads configuration only — no network call. Startup
    should not depend on an external service being reachable, and the failure
    this actually needs to catch is local: tracing switched on with no API key,
    which otherwise produces no traces and no error.

    The index checks are the RAG-specific half. An empty or stale index does not
    stop the service starting; it just makes every answer "no relevant
    resolutions found", which reads as a model problem and is not one. Both are
    logged loudly at startup and reported on /health rather than raised, because
    a service that refuses to start cannot tell you why over HTTP.
    """
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true":
        if os.getenv("LANGSMITH_API_KEY"):
            logger.info(
                "LangSmith tracing enabled | project=%s",
                os.getenv("LANGSMITH_PROJECT", "default"),
            )
        else:
            logger.warning(
                "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set — "
                "no traces will be sent"
            )
    else:
        logger.info("LangSmith tracing disabled")

    logger.info(
        "Models | generation=%s | embedding=%s",
        active_model_id(),
        active_embedding_model_id(),
    )

    indexed = document_count()
    if indexed == 0:
        logger.warning(
            "Vector index is empty — every query will return no sources. "
            "Run `python -m app.ingest` before serving traffic."
        )
    else:
        logger.info("Vector index ready | documents=%d", indexed)

    problem = check_manifest()
    if problem:
        logger.warning("Index mismatch | %s", problem)

    # The analytics log is a side channel, not a dependency. If the database is
    # unreachable the service should still answer queries — it just will not be
    # able to tell you afterwards what it answered.
    try:
        init_db()
    except Exception as e:
        logger.warning("Analytics database unavailable: %s — search logging disabled", e)

    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Retrieves semantically similar resolved support tickets and summarises "
        "how they were fixed, grounded in the retrieved tickets and cited by ID."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post(
    "/api/v1/search",
    response_model=PTKSearchResponse,
    summary="Find and summarise how similar issues were resolved before",
    tags=["PTK"],
)
async def search_past_tickets(request: PTKSearchRequest) -> PTKSearchResponse:
    """
    Accepts a description of a live issue and returns a grounded summary of how
    similar issues were resolved, with the source tickets it drew on.

    - Input validated automatically by Pydantic (422 on invalid input)
    - Query is embedded, matched against the ticket index, and the retrieved
      tickets are summarised by the LLM
    - Sources are returned from the same retrieval that produced the answer, so
      a citation can never point at a document the model did not see
    """
    start = time.perf_counter()

    logger.info(
        f"PTK search | ticket_id={request.ticket_id or '-'} | "
        f"top_k={request.top_k or settings.retrieval_top_k} | "
        f"query_length={len(request.query)}"
    )

    try:
        result, run_id = await ainvoke_ptk_traced(
            query=request.query,
            ticket_id=request.ticket_id,
            priority=request.priority.value if request.priority else None,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=503, detail=f"Retrieval or LLM service unavailable: {e}")

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        f"PTK response | ticket_id={request.ticket_id or '-'} | "
        f"sources={result['retrieved_count']} | latency={latency_ms}ms | run_id={run_id}"
    )

    _log_search(request, result, latency_ms, run_id)

    return PTKSearchResponse(
        query=request.query,
        answer=result["answer"],
        sources=[RetrievedTicket(**source) for source in result["sources"]],
        retrieved_count=result["retrieved_count"],
        latency_ms=latency_ms,
        model_used=active_model_id(),
        embedding_model=active_embedding_model_id(),
        langsmith_run_id=run_id,
    )


def _log_search(
    request: PTKSearchRequest,
    result: dict,
    latency_ms: float,
    run_id: str | None,
) -> None:
    """
    Persist one search to the analytics log — best effort, never fatal.

    Every failure here is swallowed with a warning. The answer has already been
    generated and paid for; losing an analytics row is an inconvenience, and
    turning it into a 500 would mean the observability layer is the thing taking
    the service down.

    The write is synchronous. A local SQLite insert is well under a millisecond
    and the request has just spent seconds waiting on a model, so the cost is
    noise. Against a remote database this belongs on a queue — the honest
    threshold is when the insert stops being invisible next to the model call.
    """
    if not settings.log_searches:
        return

    try:
        with session_scope() as session:
            analytics.log_search(
                session,
                query=request.query,
                generation_model=active_model_id(),
                embedding_model=active_embedding_model_id(),
                top_k=request.top_k or settings.retrieval_top_k,
                latency_ms=latency_ms,
                sources=result["sources"],
                ticket_id=request.ticket_id,
                priority=request.priority.value if request.priority else None,
                langsmith_run_id=run_id,
                prompt_tokens=result.get("prompt_tokens"),
                response_tokens=result.get("response_tokens"),
            )
    except Exception as e:
        logger.warning(f"Search logging failed (answer was returned anyway): {e}")


@app.get(
    "/api/v1/analytics/usage",
    response_model=list[ModelUsageRow],
    summary="Calls, tokens, cost and latency per model",
    tags=["Analytics"],
)
def usage_analytics(db: Session = Depends(get_db_session)) -> list[ModelUsageRow]:
    """
    Aggregate spend and latency per generation model.

    `unpriced_calls` is the caveat column: providers that do not report token
    usage produce NULL cost, which SUM skips. Without that count the total looks
    complete when it may be covering a third of the traffic.
    """
    return [ModelUsageRow(**row) for row in analytics.usage_by_model(db)]


@app.get(
    "/api/v1/analytics/latency",
    response_model=list[LatencyPercentileRow],
    summary="Latency at a percentile, per model",
    tags=["Analytics"],
)
def latency_analytics(
    percentile: float = Query(default=0.95, gt=0.0, lt=1.0),
    db: Session = Depends(get_db_session),
) -> list[LatencyPercentileRow]:
    """
    Computed with window functions, because SQLite has no percentile aggregate.

    Percentiles rather than averages: a 900ms mean with a 6s p95 is a service
    that feels broken to one user in twenty, and the mean will never say so.
    """
    return [
        LatencyPercentileRow(**row)
        for row in analytics.latency_percentile_by_model(db, percentile)
    ]


@app.get(
    "/api/v1/analytics/retrieval-coverage",
    response_model=RetrievalCoverageResponse,
    summary="Which indexed tickets are actually being retrieved",
    tags=["Analytics"],
)
def retrieval_coverage(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db_session),
) -> RetrievalCoverageResponse:
    """
    The RAG-specific analytic: how much of the index earns its place.

    A ticket that never surfaces is either irrelevant to what people ask, or
    written in language nobody uses — the second is a fixable retrieval problem
    disguised as a content problem, and this is the only place it shows up.
    """
    most = analytics.top_retrieved_tickets(db, limit=limit)
    never = analytics.never_retrieved_tickets(db)

    indexed = db.query(analytics.CorpusTicket).count()
    retrieved_once = max(indexed - len(never), 0)
    coverage = round(100.0 * retrieved_once / indexed, 1) if indexed else 0.0

    return RetrievalCoverageResponse(
        indexed_tickets=indexed,
        retrieved_at_least_once=retrieved_once,
        coverage_pct=coverage,
        most_retrieved=most,
        never_retrieved=never,
    )


@app.get(
    "/api/v1/analytics/volume",
    response_model=list[DailyVolumeRow],
    summary="Searches per day with day-over-day change",
    tags=["Analytics"],
)
def volume_analytics(db: Session = Depends(get_db_session)) -> list[DailyVolumeRow]:
    """Uses LAG to read the previous day's count without a self-join."""
    return [DailyVolumeRow(**row) for row in analytics.daily_volume_with_growth(db)]


@app.get(
    "/api/v1/analytics/slowest",
    response_model=list[SlowSearchRow],
    summary="Slowest searches per model",
    tags=["Analytics"],
)
def slowest_analytics(
    n: int = Query(default=3, ge=1, le=20, description="Rows per model"),
    db: Session = Depends(get_db_session),
) -> list[SlowSearchRow]:
    """
    Top-N per group via ROW_NUMBER() OVER (PARTITION BY ...).

    Each row carries its `langsmith_run_id`: this endpoint finds the slow
    requests, and that ID opens the trace that explains them.
    """
    return [SlowSearchRow(**row) for row in analytics.slowest_searches_per_model(db, n)]


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check() -> HealthResponse:
    """
    Health check for load balancer and monitoring.

    Reports index state as well as liveness. A RAG service with an empty index is
    up by every ordinary measure and useless, so "healthy" here means the process
    is running *and* has something to retrieve from.
    """
    indexed = document_count()
    warning = check_manifest()

    if indexed == 0:
        warning = warning or (
            "vector index is empty — run `python -m app.ingest`"
        )

    return HealthResponse(
        status="healthy" if indexed > 0 and warning is None else "degraded",
        version=settings.app_version,
        indexed_documents=indexed,
        index_warning=warning,
    )
