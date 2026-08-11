"""
Tests for the retrieval half of the pipeline.

These assert against a real Chroma index built by the session fixture, not a
mocked retriever. Mocking retrieval in a RAG project tests the wiring and skips
the part that actually decides answer quality — and every one of these would
still pass if the embedding model were returning constants.
"""

from app.config import settings
from app.corpus import HISTORICAL_TICKETS
from app.embeddings import KeywordEmbeddings
from app.ptk_chain import NO_RESULTS_CONTEXT, format_docs, to_sources
from app.vectorstore import check_manifest, document_count, get_retriever
from langchain_core.documents import Document


def _retrieved_ids(query: str, k: int | None = None) -> list[str]:
    documents = get_retriever(k).invoke(query)
    return [doc.metadata["ticket_id"] for doc in documents]


def test_corpus_is_indexed():
    assert document_count() == len(HISTORICAL_TICKETS)


def test_manifest_matches_active_embedding_model():
    assert check_manifest() is None


def test_ticket_ids_are_unique():
    """
    Chroma upserts by ID, so a duplicate ID in the corpus silently drops a
    ticket rather than raising — the collection just ends up one document short.
    """
    ids = [ticket.ticket_id for ticket in HISTORICAL_TICKETS]
    assert len(ids) == len(set(ids))


def test_resolution_is_part_of_embedded_text():
    """
    Guards the decision in corpus.py: the resolution has to be inside the
    document text, not only in metadata. If it moves to metadata, the model is
    asked to ground its answer in text that never contained one, and RAGAS
    faithfulness collapses for a reason that looks like a model fault.
    """
    ticket = HISTORICAL_TICKETS[0]
    text = ticket.to_document_text()
    assert ticket.resolution[:40] in text
    assert ticket.issue[:40] in text


def test_database_query_retrieves_database_ticket():
    assert "HIST-009" in _retrieved_ids(
        "connection pool exhausted, max_connections exceeded, batch will not start"
    )


def test_as2_query_retrieves_as2_ticket():
    assert "HIST-003" in _retrieved_ids(
        "AS2 MDN receipts never arrive, messages unacknowledged by partner"
    )


def test_top_k_is_respected():
    assert len(_retrieved_ids("database connection pool exhausted", k=1)) == 1
    assert len(_retrieved_ids("database connection pool exhausted", k=5)) == 5


def test_retriever_defaults_to_configured_top_k():
    assert len(_retrieved_ids("api rate limit throttling")) == settings.retrieval_top_k


def test_unrelated_query_scores_below_topical_query():
    """
    A k-nearest retriever always returns k documents, so an out-of-domain query
    still gets results — the guard is that they are measurably worse matches, not
    that there are none. This is why the prompt has to license refusal and why
    RETRIEVAL_SCORE_THRESHOLD exists.
    """
    store = get_retriever().vectorstore
    on_topic = store.similarity_search_with_score(
        "database connection pool exhausted", k=1
    )
    off_topic = store.similarity_search_with_score(
        "recommended seating plan for the office move", k=1
    )
    # Chroma returns distance, so lower is a closer match.
    assert on_topic[0][1] < off_topic[0][1]


def test_keyword_embeddings_are_deterministic_across_instances():
    """
    Ingestion and querying happen in different processes. Python's hash() is
    salted per process, so an embedding built on it would put the same text in
    two different places — this is the regression test for that bug.
    """
    first = KeywordEmbeddings().embed_query("database connection pool exhausted")
    second = KeywordEmbeddings().embed_query("database connection pool exhausted")
    assert first == second


def test_keyword_embeddings_place_related_text_closer():
    embeddings = KeywordEmbeddings()

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))   # both are unit vectors

    query = embeddings.embed_query("database connection pool exhausted")
    related = embeddings.embed_documents(["SQL connection pool max_connections exceeded"])[0]
    unrelated = embeddings.embed_documents(["AS2 MDN receipt certificate expired"])[0]

    assert cosine(query, related) > cosine(query, unrelated)


def test_format_docs_labels_documents_with_ticket_ids():
    context = format_docs([
        Document(
            page_content="Issue: pool exhausted\nResolution: raised the ceiling",
            metadata={"ticket_id": "HIST-009", "category": "Database", "partner": "Acme Corp"},
        )
    ])
    assert "[HIST-009]" in context
    assert "raised the ceiling" in context


def test_format_docs_handles_empty_retrieval():
    assert format_docs([]) == NO_RESULTS_CONTEXT


def test_to_sources_truncates_long_excerpts():
    sources = to_sources(
        [Document(page_content="x" * 500, metadata={"ticket_id": "HIST-001"})],
        excerpt_chars=100,
    )
    assert sources[0]["excerpt"].endswith("...")
    assert len(sources[0]["excerpt"]) == 103
