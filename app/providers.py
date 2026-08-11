"""
Provider selection and retry policy, shared by the chain, the embeddings and the
eval judge.

One setting — LLM_PROVIDER — governs all three. A judge pinned to one provider
would make the eval harness unrunnable in exactly the environments the fallback
exists for, and embeddings resolved independently of the chat model would let a
machine end up embedding with Titan while generating with a local model, which
is confusing rather than clever.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import settings

logger = logging.getLogger(__name__)


# The canned answer for the "fake" provider. Written to read like a real
# summary of retrieved resolutions so eval criteria have something meaningful to
# check — but a score against this measures the harness, not the model.
_FAKE_ANSWER = (
    "Based on the historical resolutions retrieved, this pattern has been caused "
    "by a configuration drift on one side of the integration rather than a fault "
    "in the transport itself. The recorded fix was to compare the current partner "
    "profile against the values the connection was originally provisioned with, "
    "correct the field that changed, and replay the affected documents from the "
    "error queue. Two of the referenced tickets also added a validation step so "
    "the same drift fails loudly next time instead of silently."
)


def _auto_detect_provider() -> str:
    """
    Pick a provider by probing for AWS credentials.

    Credentials are resolved through boto3 itself rather than by checking AWS_*
    environment variables: `aws configure` writes to ~/.aws/credentials and sets
    no env vars, so an env-var check reports "no AWS" on the most common local
    setup. Kept separate from resolve_provider so tests can stub the probe —
    calling it without credentials falls through to the EC2 metadata endpoint and
    blocks until that times out.
    """
    try:
        import boto3

        if boto3.Session().get_credentials() is not None:
            return "bedrock"
        logger.info("No AWS credentials resolvable — using ollama")
    except Exception as e:  # pragma: no cover - depends on local install
        logger.info(f"boto3 unavailable ({e}) — using ollama")

    return "ollama"


def resolve_provider() -> str:
    """Decide which provider to use. An explicit setting always wins."""
    configured = settings.llm_provider.strip().lower()
    if configured in {"bedrock", "ollama", "fake"}:
        return configured
    if configured != "auto":
        logger.warning(f"Unknown llm_provider '{configured}' — falling back to auto")

    return _auto_detect_provider()


def active_model_id() -> str:
    """
    Identifier for the generation model actually in use, for responses and trace
    metadata.

    Reporting settings.bedrock_model_id unconditionally would label an Ollama or
    fake run as Claude 3 Sonnet, which is exactly the kind of thing observability
    is supposed to stop you doing.
    """
    provider = resolve_provider()
    if provider == "bedrock":
        return settings.bedrock_model_id
    if provider == "ollama":
        return f"ollama:{settings.ollama_model}"
    return "fake:canned-answer"


def active_embedding_model_id() -> str:
    """
    Identifier for the embedding model in use.

    Recorded in trace metadata and written into the index manifest at ingestion
    time, so a mismatch between the model that built the index and the model
    querying it is visible rather than inferred from bad results.
    """
    provider = resolve_provider()
    if provider == "bedrock":
        return settings.bedrock_embedding_model_id
    if provider == "ollama":
        return f"ollama:{settings.ollama_embedding_model}"
    return "fake:keyword-embeddings"


def build_chat_model(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    bedrock_model_id: str | None = None,
    ollama_model: str | None = None,
    fake_responses: list[str] | None = None,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Construct a chat model for the resolved provider (providers imported lazily).

    Overrides let a caller keep the provider choice while changing the model or
    sampling — the judge wants temperature 0 and a cheaper model than the chain.
    """
    provider = resolve_provider()
    temperature = settings.bedrock_temperature if temperature is None else temperature
    max_tokens = settings.bedrock_max_tokens if max_tokens is None else max_tokens

    if provider == "bedrock":
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=bedrock_model_id or settings.bedrock_model_id,
            region_name=settings.aws_default_region,
            model_kwargs={"max_tokens": max_tokens, "temperature": temperature},
            streaming=streaming,   # enables chain.astream()
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # No streaming flag: ChatOllama streams through .astream() natively.
        return ChatOllama(
            model=ollama_model or settings.ollama_model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    logger.warning(
        "llm_provider=fake — the answer is canned. Retrieval, tracing and evals "
        "are real; answer quality is not."
    )
    return FakeListChatModel(responses=fake_responses or [_FAKE_ANSWER])


# ─────────────────────────────────────────────────────────────────────────────
# Retry
# ─────────────────────────────────────────────────────────────────────────────

# Three attempts is the AWS-recommended starting point for API retries. Beyond
# that you are holding an HTTP request open long enough that the caller times out
# anyway, and the exponential backoff means attempt four alone waits ~4s.
RETRY_ATTEMPTS = 3


def transient_exception_types(provider: str | None = None) -> tuple[type[BaseException], ...]:
    """
    Exceptions worth a second attempt on this provider.

    Retrying everything is the trap: a KeyError from a missing prompt variable or
    a ValueError from bad input will fail identically three times, so the only
    thing a blanket retry buys is a slower error and a masked bug.

    Imports are local because the provider packages are optional — an
    Ollama-only install has no botocore, and vice versa.

    One honest limitation on Bedrock: botocore raises ClientError for throttling
    (retryable) and for AccessDenied (not), and with_retry filters by exception
    type only — it takes no predicate to inspect the error code. So a
    misconfigured IAM role costs three attempts, roughly three seconds, before
    the real error surfaces. That is the cheaper side of the trade: the
    alternative is not retrying throttles at all.
    """
    provider = provider or resolve_provider()

    if provider == "bedrock":
        from botocore.exceptions import ClientError
        from botocore.exceptions import ConnectionError as BotoConnectionError

        # BotoConnectionError is the base for EndpointConnectionError,
        # ConnectTimeoutError and ReadTimeoutError.
        return (ClientError, BotoConnectionError)

    if provider == "ollama":
        import httpx

        # httpx.HTTPError covers ConnectError and the timeout family — the way a
        # local Ollama daemon actually fails: not running yet, or still loading a
        # model into memory. ollama.ResponseError is deliberately excluded; it is
        # also raised for "model not found", which no amount of retrying fixes.
        return (httpx.HTTPError, ConnectionError, TimeoutError)

    # fake: nothing to be transient about.
    return ()


def with_transient_retry(runnable: Runnable) -> Runnable:
    """
    Wrap a runnable so transient provider failures are retried with backoff.

    Exponential backoff with jitter — 1s, 2s, 4s plus a random offset. The jitter
    matters under load: without it, every request throttled in the same second
    retries in the same second, and the burst that caused the throttle is
    reproduced exactly.

    Applied per step rather than to the whole chain. In this pipeline that
    distinction has teeth: retrying the chain would re-run retrieval — a second
    embedding call and a second vector search — to recover from a throttled
    generation call that has nothing to do with either.
    """
    retry_on = transient_exception_types()
    if not retry_on:
        return runnable

    return runnable.with_retry(
        retry_if_exception_type=retry_on,
        wait_exponential_jitter=True,
        stop_after_attempt=RETRY_ATTEMPTS,
    )
