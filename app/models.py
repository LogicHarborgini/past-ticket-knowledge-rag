"""
Pydantic models for the PTK API.

These define the data contracts:
- PTKSearchRequest:  what the API expects as input
- PTKSearchResponse: what the API returns
- RetrievedTicket:   one cited source behind an answer
- HealthResponse:    for the /health endpoint

FastAPI uses these for automatic validation and OpenAPI docs generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TicketPriority(str, Enum):
    """Priority of the ticket the engineer is working on — only these values."""
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class PTKSearchRequest(BaseModel):
    """Request body for POST /api/v1/search."""

    # No min_length on the field: the validator below strips first, so it
    # rejects "          " which a field constraint would let through.
    query: str = Field(..., description="Natural-language description of the current issue")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Historical tickets to retrieve. Defaults to RETRIEVAL_TOP_K. Capped "
            "at 10 because every extra document spends context budget the answer "
            "has to share — past a handful, precision falls faster than recall rises."
        ),
    )
    # Both optional and unused by the chain — they exist to be written into the
    # trace, which is what makes a run findable later by the ticket it belonged to.
    ticket_id: Optional[str] = Field(
        default=None,
        description="ID of the live ticket this search relates to, for trace correlation",
    )
    priority: Optional[TicketPriority] = Field(default=None)

    @field_validator("query")
    @classmethod
    def query_must_be_meaningful(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError(
                f"Query too short ({len(stripped)} chars). Min 10 required — a "
                "two-word query embeds to a vector too vague to retrieve against."
            )
        return stripped

    @field_validator("ticket_id")
    @classmethod
    def normalise_ticket_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = v.strip().upper()
        return stripped or None

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Production database connection pool is exhausted and the batch job cannot start",
                "top_k": 3,
                "ticket_id": "tick-12345",
                "priority": "P1",
            }
        }
    }


class RetrievedTicket(BaseModel):
    """
    One historical ticket the answer was grounded in.

    Returned with every answer on purpose. An unsourced summary asks the engineer
    to trust it; a sourced one lets them open HIST-009 and check. That is the
    difference between a RAG system people use and one they stop using after it
    is confidently wrong once.
    """

    ticket_id: str
    partner: Optional[str] = None
    category: Optional[str] = None
    excerpt: str = Field(..., description="Opening of the retrieved ticket text")


class PTKSearchResponse(BaseModel):
    """Response body from POST /api/v1/search."""

    query: str
    answer: str
    sources: list[RetrievedTicket]
    retrieved_count: int
    latency_ms: float
    model_used: str
    embedding_model: str
    status: str = "success"
    # LangSmith trace ID for this run. None when tracing is disabled. Returning
    # it lets a query in your own records be matched to its trace afterwards.
    langsmith_run_id: Optional[str] = Field(default=None)


class ModelUsageRow(BaseModel):
    """One row of GET /api/v1/analytics/usage."""

    generation_model: str
    calls: int
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    avg_latency_ms: float
    max_latency_ms: float
    # How many of those calls could not be priced. A cost total without this is
    # unreadable: two unpriced calls out of three makes the total meaningless,
    # and nothing else on the row would reveal it.
    unpriced_calls: int


class LatencyPercentileRow(BaseModel):
    """One row of GET /api/v1/analytics/latency."""

    generation_model: str
    percentile: float
    latency_ms: float
    samples: int


class RetrievedTicketStats(BaseModel):
    """One row of the retrieved-ticket leaderboard."""

    ticket_id: str
    category: Optional[str] = None
    times_retrieved: int
    times_ranked_first: int
    avg_rank: float


class UnretrievedTicket(BaseModel):
    """An indexed ticket no search has ever returned."""

    ticket_id: str
    category: Optional[str] = None
    partner: Optional[str] = None
    indexed_at: Optional[str] = None


class RetrievalCoverageResponse(BaseModel):
    """Response body from GET /api/v1/analytics/retrieval-coverage."""

    indexed_tickets: int
    retrieved_at_least_once: int
    coverage_pct: float
    most_retrieved: list[RetrievedTicketStats]
    never_retrieved: list[UnretrievedTicket]


class DailyVolumeRow(BaseModel):
    """One row of GET /api/v1/analytics/volume."""

    day: str
    searches: int
    previous_day: Optional[int] = None
    change_pct: Optional[float] = None


class SlowSearchRow(BaseModel):
    """One row of GET /api/v1/analytics/slowest."""

    generation_model: str
    query: str
    latency_ms: float
    retrieved_count: int
    # The join key back to LangSmith: this query finds the slow requests, that ID
    # opens the trace explaining them.
    langsmith_run_id: Optional[str] = None
    called_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body from GET /health."""

    status: str = "healthy"
    service: str = "PTK"
    version: str
    # An empty index is the failure this endpoint exists to catch. The service
    # starts fine without one and answers every query with "no relevant
    # resolutions found", which looks like a model problem and is not.
    indexed_documents: int = 0
    index_warning: Optional[str] = None
