"""
Tests for chain assembly and the traced entry point.

Everything here runs on LLM_PROVIDER=fake, so the assertions are about the shape
of the pipeline — what it retrieves, what it returns, what it records — never
about the wording of a generated answer. Asserting on model prose is how a test
suite becomes a thing people delete.
"""

import pytest

from app.providers import (
    active_embedding_model_id,
    active_model_id,
    resolve_provider,
    transient_exception_types,
    with_transient_retry,
)
from app.ptk_chain import _cached_chat_model, get_ptk_chain, ainvoke_ptk_traced


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """The chat model is cached per process; tests should not inherit each other's."""
    _cached_chat_model.cache_clear()
    yield
    _cached_chat_model.cache_clear()


def test_provider_is_fake_under_test():
    assert resolve_provider() == "fake"


def test_model_ids_report_the_provider_actually_in_use():
    """
    Reporting the configured Bedrock model ID while running on something else is
    the exact failure observability is supposed to prevent — a dashboard full of
    Claude latencies that were never Claude.
    """
    assert active_model_id() == "fake:canned-answer"
    assert active_embedding_model_id() == "fake:keyword-embeddings"


def test_fake_provider_has_no_retryable_exceptions():
    assert transient_exception_types("fake") == ()


def test_with_transient_retry_is_a_noop_without_retryable_types():
    """A retry wrapper around a provider with nothing transient adds a layer for
    no benefit, so the wrapper returns the runnable untouched."""
    chain = get_ptk_chain()
    assert with_transient_retry(chain) is chain


@pytest.mark.asyncio
async def test_chain_returns_question_documents_and_answer():
    result = await get_ptk_chain().ainvoke(
        "connection pool exhausted and the nightly batch will not start"
    )
    assert set(result) == {"question", "documents", "answer"}
    assert result["question"].startswith("connection pool")
    assert result["documents"]
    assert isinstance(result["answer"], str) and result["answer"]


@pytest.mark.asyncio
async def test_sources_come_from_the_same_retrieval_as_the_answer():
    """
    The reason the chain returns documents rather than the API re-querying for
    citations: a second retrieval is a second query, and nothing guarantees it
    returns what the model actually read.
    """
    chain_result = await get_ptk_chain().ainvoke("AS2 MDN receipts are missing")
    payload, _run_id = await ainvoke_ptk_traced(query="AS2 MDN receipts are missing")

    from_chain = [doc.metadata["ticket_id"] for doc in chain_result["documents"]]
    from_traced = [source["ticket_id"] for source in payload["sources"]]
    assert from_chain == from_traced


@pytest.mark.asyncio
async def test_traced_invocation_returns_payload_and_no_run_id_when_tracing_off():
    payload, run_id = await ainvoke_ptk_traced(
        query="Database connection pool exhausted on production",
        ticket_id="TICK-1",
        priority="P1",
    )
    assert payload["retrieved_count"] == len(payload["sources"])
    assert payload["answer"]
    # conftest sets LANGSMITH_TRACING=false: there is no trace to point at, so
    # the ID must be None rather than a fabricated UUID.
    assert run_id is None


@pytest.mark.asyncio
async def test_top_k_override_reaches_the_retriever():
    payload, _ = await ainvoke_ptk_traced(
        query="Database connection pool exhausted on production", top_k=1
    )
    assert payload["retrieved_count"] == 1


@pytest.mark.asyncio
async def test_sources_carry_citable_metadata():
    payload, _ = await ainvoke_ptk_traced(query="EDI 850 rejected with X12-834")
    for source in payload["sources"]:
        assert source["ticket_id"].startswith("HIST-")
        assert source["excerpt"]
