"""
Embedding model selection.

Production embeds with Amazon Titan; a machine without AWS falls back to a local
Ollama model; `LLM_PROVIDER=fake` uses the keyword embeddings below.

On embed_query vs embed_documents: Titan v2 and nomic-embed-text are trained
asymmetrically — a query and a passage carrying the same meaning are encoded
with different instructions so that queries land near the passages that answer
them, not near other queries. Both LangChain wrappers expose the two methods,
and the vector store calls the right one on each side. Nothing here has to
choose; the point of writing it down is that using the wrong method by hand at
ingestion time degrades retrieval without raising anything.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.providers import resolve_provider

logger = logging.getLogger(__name__)


class KeywordEmbeddings(Embeddings):
    """
    Deterministic, dependency-free embeddings for the `fake` provider.

    This is not a semantic model and does not pretend to be one. It maps a fixed
    vocabulary of integration-support terms onto topic dimensions, adds a hashed
    residual for everything else, and L2-normalises. That is enough for the
    property the offline path actually needs: a query about database connection
    pools retrieves the database tickets and not the AS2 ones, every run, with no
    model server and no network.

    What it buys is a test suite and a demo that assert real retrieval behaviour
    rather than mocking the retriever away. What it cannot do is generalise —
    'the pipe is blocked' will not find the connection-pool ticket, because there
    is no learned notion that those are related. Real embeddings are what make
    that work, and swapping them in is a provider setting.
    """

    # Topic buckets. Terms sharing a bucket are treated as the same axis, which
    # is the one property of a real embedding space worth simulating: different
    # words, same direction.
    TOPIC_TERMS: dict[int, tuple[str, ...]] = {
        0: ("edi", "x12", "850", "810", "translator", "envelope", "edifact"),
        1: ("as2", "mdn", "receipt", "acknowledgement", "acknowledgment"),
        2: ("sftp", "ftp", "ssh", "file", "files", "directory", "transfer", "poller", "pickup"),
        3: ("api", "rest", "endpoint", "webhook", "gateway", "http", "request", "requests"),
        4: ("database", "db", "sql", "query", "queries", "postgres", "postgresql", "table"),
        5: ("auth", "authentication", "login", "credential", "credentials", "jwt",
            "token", "session", "sessions", "401", "unauthorized", "password"),
        6: ("deployment", "deploy", "release", "rollback", "migration", "pipeline", "staging"),
        7: ("network", "firewall", "whitelist", "timeout", "timing", "latency", "ip", "port"),
        8: ("certificate", "cert", "tls", "ssl", "expired", "signing", "rotation", "rotated"),
        9: ("throttle", "throttled", "throttling", "rate", "limit", "429", "backoff"),
        10: ("pool", "connection", "connections", "max_connections", "exhausted", "leak"),
        11: ("slow", "performance", "index", "scan", "degraded", "seconds"),
        12: ("queue", "queued", "batch", "job", "scheduled", "replay", "backed"),
        13: ("partner", "profile", "configuration", "config", "mapping", "map", "settings"),
        14: ("error", "errors", "failure", "failing", "failed", "rejected", "reject", "broken"),
        15: ("memory", "redis", "cache", "eviction", "store"),
    }

    TOPIC_DIMS = 16
    HASH_DIMS = 8
    DIMENSION = TOPIC_DIMS + HASH_DIMS

    # Topic hits count for more than hashed residual, so a shared subject beats
    # incidental shared vocabulary. Without this an out-of-domain query would
    # score respectably against everything through common English alone.
    TOPIC_WEIGHT = 1.0
    HASH_WEIGHT = 0.25

    def __init__(self) -> None:
        # Inverted once at construction: term -> dimension.
        self._term_to_dim = {
            term: dim
            for dim, terms in self.TOPIC_TERMS.items()
            for term in terms
        }

    def _vectorise(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSION

        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            dim = self._term_to_dim.get(token)
            if dim is not None:
                vector[dim] += self.TOPIC_WEIGHT
                continue

            if len(token) < 4:
                # Stop-word proxy. Short tokens are overwhelmingly 'the', 'and',
                # 'was' — letting them into the residual makes every document
                # drift toward the same corner of the space.
                continue

            # md5 for a stable bucket across processes and platforms. Python's
            # hash() is salted per process, which would make the same text embed
            # differently between the ingest run and the query run.
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = self.TOPIC_DIMS + (digest[0] % self.HASH_DIMS)
            vector[bucket] += self.HASH_WEIGHT

        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude == 0.0:
            # No recognised content at all. A zero vector is undefined under
            # cosine similarity, so return a fixed off-axis unit vector instead —
            # far from every real document, which is the correct answer here.
            vector[-1] = 1.0
            return vector

        return [v / magnitude for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorise(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorise(text)


# Set once the fake-provider warning has been emitted. build_embeddings() is
# called on every vector store handle — several times per request — and an
# unconditional warning buries the log it is meant to stand out in.
_warned_about_fake = False


def build_embeddings() -> Embeddings:
    """Construct the embedding model for the resolved provider."""
    global _warned_about_fake
    provider = resolve_provider()

    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model_id,
            region_name=settings.aws_default_region,
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.ollama_embedding_model)

    if not _warned_about_fake:
        logger.warning(
            "llm_provider=fake — using keyword embeddings. Retrieval is deterministic "
            "and topically sane, but there is no semantic generalisation."
        )
        _warned_about_fake = True

    return KeywordEmbeddings()
