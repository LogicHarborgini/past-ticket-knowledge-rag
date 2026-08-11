"""
The PTK RAG chain.

    query ──▶ retriever ──▶ format_docs ──▶ prompt ──▶ llm ──▶ parser
                   │                                              │
                   └──────────── documents ───────────────────────┴──▶ sources

Retrieval and generation are two failure modes wearing one coat. When an answer
is wrong it is either because the wrong tickets were retrieved or because the
model ignored the right ones, and those have opposite fixes. Everything in this
module is arranged so a trace tells you which one happened: the retrieved
documents are a visible span, the formatted context is a visible span, and the
sources come back on the response so a bad answer can be checked against what it
was actually given.
"""

from __future__ import annotations

import logging
import os
import uuid
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langsmith import traceable

from app.config import settings
from app.providers import (
    active_embedding_model_id,
    active_model_id,
    build_chat_model,
    with_transient_retry,
)
from app.vectorstore import get_retriever

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

# Three instructions here exist to move a specific measured score, not to sound
# thorough:
#
#   "only from the resolutions below"  -> faithfulness. Without it the model
#       supplements retrieved tickets with generic troubleshooting advice that
#       reads plausibly and is not grounded in anything.
#   "say so plainly"                   -> the out-of-domain case. A retriever
#       always returns its k nearest neighbours, however far away they are; if
#       the prompt does not license refusal, the model invents a connection to
#       whatever came back.
#   "cite the ticket IDs"              -> makes the grounding checkable by the
#       engineer reading it, and by the eval harness.
SYSTEM_PROMPT = """You are an enterprise integration support engineer helping a \
colleague resolve a live ticket by drawing on how similar issues were resolved \
before.

Work only from the historical resolutions provided below. Do not add \
troubleshooting steps from your own general knowledge, however reasonable they \
seem — an unsupported suggestion is worse than a short answer, because the \
engineer cannot tell the two apart.

If the provided resolutions do not address the current issue, say so plainly \
and do not offer a substitute.

Cite the ticket IDs you drew each step from. Keep the summary under 150 words \
and lead with the action most likely to resolve the issue."""

HUMAN_TEMPLATE = """Historical resolutions:
{context}

Current issue:
{question}

Summarise the relevant past resolutions:"""

NO_RESULTS_CONTEXT = "No relevant historical resolutions were found for this query."


# ─────────────────────────────────────────────────────────────────────────────
# Context formatting
# ─────────────────────────────────────────────────────────────────────────────


@traceable(name="format-retrieved-tickets", tags=["retrieval", "formatting"])
def format_docs(documents: list[Document]) -> str:
    """
    Turn retrieved Documents into the context string — the "stuffing" step.

    Traced because this is the exact text the model sees. When an answer is
    wrong, the first question is whether the context contained the answer at all,
    and reconstructing that from the retriever's output afterwards is guesswork.
    Here it is a span you can open.

    Documents are labelled with their ticket ID so the model has something real
    to cite. Numbering them "Resolution 1, 2, 3" instead would give it only
    positions to cite, and positions mean nothing to the engineer reading the
    answer.
    """
    if not documents:
        return NO_RESULTS_CONTEXT

    blocks = []
    for index, doc in enumerate(documents, start=1):
        ticket_id = doc.metadata.get("ticket_id", f"UNKNOWN-{index}")
        category = doc.metadata.get("category", "uncategorised")
        partner = doc.metadata.get("partner", "unknown partner")
        blocks.append(
            f"[{ticket_id}] category={category} | partner={partner}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


def _sum_usage(usage_metadata: dict[str, Any]) -> dict[str, int | None]:
    """
    Collapse per-model usage into one pair of totals.

    The callback keys usage by model name because a chain can call more than one.
    This one calls a single model, but summing rather than reading [0] means a
    future reranker or query-rewriting step is accounted for instead of silently
    uncounted.

    An empty dict means the provider reported nothing — local models often do
    not, and the fake provider never does — and that becomes None, not zero.
    """
    if not usage_metadata:
        return {"prompt_tokens": None, "response_tokens": None}

    prompt = sum(u.get("input_tokens", 0) for u in usage_metadata.values())
    response = sum(u.get("output_tokens", 0) for u in usage_metadata.values())
    return {"prompt_tokens": prompt or None, "response_tokens": response or None}


def to_sources(documents: list[Document], excerpt_chars: int = 240) -> list[dict[str, Any]]:
    """
    Reduce Documents to the citation payload returned with the answer.

    Excerpted rather than returned whole: the caller needs enough to recognise
    the ticket and a stable ID to look it up with, not a second copy of the
    corpus in every HTTP response.
    """
    sources = []
    for doc in documents:
        content = doc.page_content.strip()
        excerpt = content[:excerpt_chars] + ("..." if len(content) > excerpt_chars else "")
        sources.append(
            {
                "ticket_id": doc.metadata.get("ticket_id", "UNKNOWN"),
                "partner": doc.metadata.get("partner"),
                "category": doc.metadata.get("category"),
                "excerpt": excerpt,
            }
        )
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Chain
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _cached_chat_model() -> BaseChatModel:
    """
    Build the chat model once per process.

    Only the model is cached, unlike the sibling first-response service which
    caches the whole chain. The chain here closes over a Chroma handle, and a
    cached chain would keep serving a collection that ingestion has since
    rebuilt — stale results with no error to explain them. Constructing a
    retriever is microseconds; constructing a boto3 session is not, so this
    caches the part that is actually expensive.
    """
    return build_chat_model()


def get_ptk_chain(top_k: int | None = None) -> Runnable:
    """
    Build the RAG chain.

    Returns
    -------
    Runnable
        Takes a query string, returns
        {"question": str, "documents": list[Document], "answer": str}.

    Returning the documents alongside the answer is what lets the API cite its
    sources from the same retrieval that produced the answer. The obvious
    alternative — run `retriever | format_docs | prompt | llm | parser`, then
    call the retriever a second time to collect sources for display — doubles
    the embedding calls and, worse, can cite documents the answer never saw:
    the two retrievals are separate queries and nothing guarantees they agree.
    """
    retriever = with_transient_retry(get_retriever(top_k))
    llm = with_transient_retry(_cached_chat_model())

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    # Consumes {"question", "documents"} and produces the answer string. The
    # prompt reads "question" and "context" and ignores the extra "documents"
    # key it is handed.
    answer_chain: Runnable = (
        RunnablePassthrough.assign(context=lambda payload: format_docs(payload["documents"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    # RunnableParallel fans the query out: one branch retrieves, the other passes
    # the question through untouched for the prompt. RunnablePassthrough.assign
    # then adds the answer without discarding the documents.
    return (
        RunnableParallel(question=RunnablePassthrough(), documents=retriever)
        | RunnablePassthrough.assign(answer=answer_chain)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Traced invocation
# ─────────────────────────────────────────────────────────────────────────────


def _tracing_enabled() -> bool:
    """Whether LangSmith tracing is switched on for this process."""
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


@traceable(run_type="chain")
async def _ptk_pipeline(
    *,
    query: str,
    ticket_id: str | None,
    priority: str | None,
    top_k: int | None,
) -> dict[str, Any]:
    """
    One end-to-end PTK search: retrieve, ground, generate.

    Wrapping the whole operation in a single @traceable parent is what gives the
    trace tree its shape. A @traceable function called outside any active trace
    becomes its own root run, so format-retrieved-tickets would otherwise appear
    as an orphan trace sitting next to the search rather than a span inside it:

        PTK-TICK-5001
        ├── RunnableParallel                  [210ms]
        │   ├── VectorStoreRetriever          [205ms]  ← what was retrieved
        │   └── RunnablePassthrough           [0ms]
        ├── format-retrieved-tickets          [1ms]    ← what the model was given
        └── RunnableSequence                  [1.3s]
            └── ChatBedrock                   [1.3s]   ← tokens, cost, latency

    ticket_id and priority are unused in the body but kept in the signature
    because @traceable records arguments as the run's inputs — they are how the
    trace becomes searchable.
    """
    chain = get_ptk_chain(top_k)

    # Token usage is reported by the model and thrown away by StrOutputParser,
    # which returns a bare string. This callback collects it from the raw message
    # before the parser drops it — the alternative is estimating tokens from
    # characters, which is a guess that then gets multiplied by a price and
    # written into a cost report as if it were measured.
    with get_usage_metadata_callback() as usage_callback:
        result = await chain.ainvoke(query)
        usage = _sum_usage(usage_callback.usage_metadata)

    documents: list[Document] = result["documents"]
    return {
        "answer": result["answer"],
        "sources": to_sources(documents),
        "retrieved_count": len(documents),
        # None rather than 0 when the provider reports nothing. A zero would
        # average into cost and token reports as a real measurement.
        "prompt_tokens": usage["prompt_tokens"],
        "response_tokens": usage["response_tokens"],
    }


async def ainvoke_ptk_traced(
    *,
    query: str,
    ticket_id: str | None = None,
    priority: str | None = None,
    top_k: int | None = None,
) -> tuple[dict[str, Any], str | None]:
    """
    Run one PTK search, traced end to end.

    What turns an anonymous trace into a searchable one:

    - name       the trace title. Without it every trace reads
                 "RunnableSequence" and they cannot be told apart.
    - metadata   key/value pairs to filter and group traces by.
    - tags       categorical labels, e.g. priority:P1, for saved views.

    Both model IDs go into the metadata. On a RAG pipeline the embedding model is
    the one that decides what the generation model ever gets to see, so a trace
    that records only the chat model is missing the more consequential half.

    The run ID is generated here and handed to LangSmith rather than read back
    afterwards, which avoids a callback collector and guarantees the caller and
    the trace agree on the identifier.

    Returns
    -------
    tuple[dict, str | None]
        The result payload (answer, sources, retrieved_count), and the LangSmith
        run ID — None when tracing is disabled, since there is no trace to point at.
    """
    if not _tracing_enabled():
        result = await _ptk_pipeline(
            query=query, ticket_id=ticket_id, priority=priority, top_k=top_k
        )
        return result, None

    run_id = uuid.uuid4()
    trace_label = ticket_id or "adhoc"
    result = await _ptk_pipeline(
        query=query,
        ticket_id=ticket_id,
        priority=priority,
        top_k=top_k,
        langsmith_extra={
            "run_id": run_id,
            "name": f"PTK-{trace_label}",
            "metadata": {
                "ticket_id": ticket_id or "none",
                "priority": priority or "none",
                "model_id": active_model_id(),
                "embedding_model_id": active_embedding_model_id(),
                "top_k": top_k or settings.retrieval_top_k,
                "app_version": settings.app_version,
            },
            "tags": [f"priority:{priority or 'none'}", "ptk", "rag"],
        },
    )

    return result, str(run_id)
