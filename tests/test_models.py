"""Tests for the Pydantic request/response contracts."""

import pytest
from pydantic import ValidationError

from app.models import PTKSearchRequest, RetrievedTicket, TicketPriority


def test_valid_request():
    request = PTKSearchRequest(
        query="Database connection pool exhausted on production since 14:30",
        top_k=3,
        ticket_id="tick-001",
        priority=TicketPriority.P1,
    )
    assert request.ticket_id == "TICK-001"   # normalised to uppercase
    assert request.priority is TicketPriority.P1


def test_query_is_stripped():
    request = PTKSearchRequest(query="   AS2 receipts are missing entirely   ")
    assert request.query == "AS2 receipts are missing entirely"


def test_whitespace_only_query_rejected():
    """
    The reason `query` carries no min_length: a field constraint runs before the
    validator, so 15 spaces would satisfy min_length=10 and reach the embedder.
    """
    with pytest.raises(ValidationError) as exc_info:
        PTKSearchRequest(query=" " * 15)
    assert "too short" in str(exc_info.value).lower()


def test_short_query_rejected():
    with pytest.raises(ValidationError):
        PTKSearchRequest(query="db down")


def test_top_k_upper_bound_enforced():
    with pytest.raises(ValidationError):
        PTKSearchRequest(query="Database connection pool exhausted", top_k=50)


def test_top_k_defaults_to_none_so_settings_decide():
    request = PTKSearchRequest(query="Database connection pool exhausted")
    assert request.top_k is None


def test_invalid_priority_rejected():
    with pytest.raises(ValidationError):
        PTKSearchRequest(
            query="Database connection pool exhausted",
            priority="CRITICAL",   # not one of P1/P2/P3
        )


def test_blank_ticket_id_normalises_to_none():
    request = PTKSearchRequest(query="Database connection pool exhausted", ticket_id="   ")
    assert request.ticket_id is None


def test_retrieved_ticket_requires_id_and_excerpt():
    with pytest.raises(ValidationError):
        RetrievedTicket(partner="Acme Corp")
