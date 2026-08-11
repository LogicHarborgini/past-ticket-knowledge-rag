# past-ticket-knowledge-rag

> RAG pipeline that retrieves semantically similar resolved support tickets and
> summarises how they were fixed — grounded in the retrieved tickets and cited
> by ID.

Referred to throughout as **PTK** (Past Ticket Knowledge).

## Problem Statement

When a support engineer meets an unfamiliar issue, the answer usually exists —
someone resolved the same thing eight months ago and wrote it in a ticket.
Finding it means searching thousands of historical tickets by keyword, which
fails on the thing that matters most: the same problem is rarely described in the
same words twice. "Connection pool exhausted" and "max_connections exceeded" are
one issue and share no search terms.

**PTK** solves this with a RAG pipeline. The engineer's description is embedded
into a vector, semantically similar historical tickets are retrieved by nearest-
neighbour search, and an LLM summarises how those tickets were actually resolved
— citing each ticket it drew from.

**This is a RAG system, not a generation-only one.** Every answer is grounded in
retrieved documents and returns its sources. The companion project
[Smart-First-Response-system](https://github.com/LogicHarborgini/Smart-First-Response-system)
is the generation-only case: it writes a first response from the ticket in front
of it and retrieves nothing.

## Architecture

```
  Support Engineer                                  Historical Ticket Corpus
       │                                                       │
       │  "connection pool exhausted, batch will not start"    │  (indexed offline
       ▼                                                       ▼   by app/ingest.py)
 ┌────────────┐    ┌──────────────────┐    ┌────────────────────────────────┐
 │  FastAPI   │───▶│  Titan Embeddings │───▶│  Vector index (k-NN)           │
 │  POST      │    │                   │    │  Chroma locally · OpenSearch   │
 │  /v1/search│    │  query → vector   │    │  in production                 │
 └────────────┘    │  [0.12, 0.87, …]  │    │  cosine similarity → top-k     │
                   └───────────────────┘    └───────────────┬────────────────┘
                                                            │
                                            3 similar resolved tickets
                                                            │
                                                            ▼
                                            ┌────────────────────────────────┐
                                            │  format_docs()  — "stuffing"   │
                                            │  Documents → labelled context  │
                                            └───────────────┬────────────────┘
                                                            │
                                                            ▼
                                            ┌────────────────────────────────┐
                                            │  Amazon Bedrock (Claude)       │
                                            │  "Work only from the           │
                                            │   resolutions below. Cite the  │
                                            │   ticket IDs. If they do not   │
                                            │   apply, say so."              │
                                            └───────────────┬────────────────┘
                                                            │
                                                            ▼
                                              Summary + source ticket IDs
```

**Flow**

1. Engineer describes the live issue in their own words
2. The query is embedded — Titan on AWS, `nomic-embed-text` locally
3. Nearest-neighbour search returns the top-k most similar resolved tickets
4. Retrieved tickets are formatted into a labelled context block
5. Claude summarises the relevant past resolutions, grounded in that block
6. The summary is returned with the source ticket IDs it cited

**Key design decisions**

- **Resolutions are inside the embedded text, not in metadata.** It costs a
  little retrieval precision and it is what makes the answer's grounding
  measurable: an evaluator sees exactly the text the model was asked to work
  from. With resolutions hidden in metadata, RAGAS scores faithfulness against
  documents that never contained an answer, and a correct pipeline reads as a
  hallucinating one.
- **Sources come from the same retrieval as the answer.** The chain returns the
  documents alongside the generated text rather than re-querying to build a
  citation list. Two retrievals are two queries, and nothing guarantees they
  agree — a citation that names a document the model never read is worse than no
  citation.
- **Refusal is licensed in the prompt.** A k-nearest search always returns k
  documents however far away they are, so "we have nothing on this" is not
  something retrieval can express on its own.
- **Chroma stands in for OpenSearch.** Same retriever interface, no cluster to
  provision — swapping back is a change to `app/vectorstore.py` and nothing else.
- Credentials are resolved from the boto3 credential chain, never from app config.
- Transient provider failures are retried with exponential backoff and jitter.

## Quick Start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows
# .venv/bin/pip install -r requirements.txt       # macOS / Linux

cp .env.example .env          # then edit

python -m app.ingest          # build the vector index — required before serving
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs, or:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Connection pool exhausted, batch job will not start", "priority": "P1"}'
```

**No AWS account?** Set `LLM_PROVIDER=ollama` for a local model, or
`LLM_PROVIDER=fake` to run the whole pipeline with no model at all — keyword
embeddings give real, deterministic retrieval and the answer is canned. `fake`
is what the test suite uses, which is why the suite needs no credentials.

`python -m app.ingest --rebuild` after changing the embedding model or the
corpus. Vectors carry no record of the model that made them, so an index built
with Titan and queried with `nomic-embed-text` returns confident nonsense rather
than an error — `/health` reports the mismatch by comparing against a manifest
written at ingestion time.

## Project Layout

```
app/
  corpus.py        14 synthetic resolved tickets — the knowledge base
  embeddings.py    Titan / Ollama / keyword embeddings
  vectorstore.py   Chroma collection, retriever, index manifest
  ingest.py        embeds the corpus (python -m app.ingest)
  ptk_chain.py     the LCEL RAG chain and its traced entry point
  providers.py     provider selection and retry policy
  models.py        Pydantic request/response contracts
  db.py            SQLAlchemy engine and session management
  analytics.py     the search log and the SQL that reads it
  main.py          FastAPI service
evals/
  retrieval_eval.py  retrieval only — hit rate, MRR, precision@k. No LLM.
  simple_eval.py     answer shape and citations. Deterministic, no judge.
  ragas_collect.py   runs the pipeline, writes samples for scoring
  ragas_eval.py      RAGAS scoring, in a separate environment
tests/               62 tests, offline, against a real temporary index and database
```

## Core Implementation

```python
# The chain returns documents *and* answer, so citations and text come from one
# retrieval.
answer_chain = (
    RunnablePassthrough.assign(context=lambda p: format_docs(p["documents"]))
    | prompt
    | llm
    | StrOutputParser()
)

chain = (
    RunnableParallel(question=RunnablePassthrough(), documents=retriever)
    | RunnablePassthrough.assign(answer=answer_chain)
)

result = await chain.ainvoke("connection pool exhausted on production")
# {"question": ..., "documents": [Document, ...], "answer": "..."}
```

`RunnableParallel` fans the query out — one branch retrieves, the other passes
the question through untouched for the prompt. `RunnablePassthrough.assign` then
adds the answer without discarding the documents.

**Tech stack**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API layer | FastAPI (async) | Expose search as an HTTP service |
| Orchestration | LangChain LCEL | retriever → prompt → LLM → parser |
| Embeddings | Amazon Titan v2 / `nomic-embed-text` | Text → dense vector |
| Vector store | Chroma (OpenSearch k-NN in production) | Nearest-neighbour retrieval |
| Generation | Amazon Bedrock (Claude 3 Sonnet) | Summarise retrieved resolutions |
| Validation | Pydantic | Request/response schema enforcement |
| Observability | LangSmith | Trace retrieval and generation separately |
| Analytics | SQLAlchemy + SQLite/Postgres | Aggregate cost, latency and retrieval coverage |
| Evaluation | RAGAS + local harnesses | Faithfulness, relevance, retrieval quality |

## Reliability

Bedrock returns `ThrottlingException` under load and Titan has its own request
limits, so both the retriever and the model step are wrapped in three attempts
with exponential backoff and jitter.

Two deliberate choices:

- **Retry is applied per step, not to the whole chain.** Retrying the chain would
  re-run retrieval — a second embedding call and a second vector search — to
  recover from a throttled generation call that has nothing to do with either.
- **Only transient exception types are retried.** A `KeyError` from a missing
  prompt variable fails identically three times, so a blanket retry buys nothing
  but a slower error and a hidden bug. The retryable set is chosen per provider:
  botocore errors on Bedrock, connection and timeout errors on Ollama, none on
  `fake`.

Jitter is what makes this safe under concurrency. Without it every request
throttled in the same second retries in the same second, reproducing the burst
that caused the throttle.

One limitation worth stating: botocore raises `ClientError` for throttling
(retryable) and for `AccessDenied` (not), and `with_retry` filters on exception
type with no hook for the error code. A misconfigured IAM role therefore costs
three attempts before the real error surfaces — the cheaper side of the trade,
since the alternative is not retrying throttles at all.

## Observability

Every search is traced via LangSmith. Copy `.env.example` to `.env` and set:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_key_here
LANGSMITH_PROJECT=ptk-support-assistant
```

Set `LANGSMITH_TRACING=false` to disable tracing without removing the key. The
app runs identically either way — `langsmith_run_id` in the response is simply
`null`.

A RAG trace is worth more than a generation-only one because it separates two
failure modes that produce the same symptom:

```
PTK-TICK-5001
├── RunnableParallel                  [210ms]
│   ├── VectorStoreRetriever          [205ms]  ← which tickets came back
│   └── RunnablePassthrough           [0ms]
├── format-retrieved-tickets          [1ms]    ← what the model was actually given
└── RunnableSequence                  [1.3s]
    └── ChatBedrock                   [1.3s]   ← tokens, cost, latency
```

A wrong answer is either wrong retrieval or wrong generation, and those have
opposite fixes. Open the retriever span: if the tickets are irrelevant, no prompt
change will help.

Each trace carries:

| Field | Purpose |
|-------|---------|
| `run_name` = `PTK-<ticket_id>` | Identifies the trace instead of `RunnableSequence` |
| metadata: ticket ID, priority, **both** model IDs, top_k, app version | Filter and group |
| tags: `priority:P1`, `ptk`, `rag` | Saved views |
| `format-retrieved-tickets` span | The exact context string the model received |

Both model IDs are recorded because on a RAG pipeline the embedding model decides
what the generation model ever gets to see — a trace that records only the chat
model is missing the more consequential half.

Generate sample traces:

```bash
python run_ptk_traces.py
```

See [OBSERVABILITY_NOTES.md](OBSERVABILITY_NOTES.md) for what the traces showed.

## Analytics

Traces answer *what happened in this run*. They are also per-run and subject to a
retention policy. Aggregate questions — cost per model this month, whether p95
latency moved after a model change, which corpus tickets have never once been
retrieved — need a table you own.

Every search writes one row to `searches` and one per retrieved ticket to
`retrievals`, carrying its `langsmith_run_id` so a row found in SQL opens the
trace that explains it. Writes are best-effort: the answer has already been
generated and paid for, so a logging failure costs a row and nothing else.

```
GET /api/v1/analytics/usage                # calls, tokens, cost, latency per model
GET /api/v1/analytics/latency?percentile=  # p95 per model, via window functions
GET /api/v1/analytics/retrieval-coverage   # which indexed tickets ever surface
GET /api/v1/analytics/volume               # searches per day with DoD change
GET /api/v1/analytics/slowest?n=3          # top-N slowest per model, with run IDs
```

`retrieval-coverage` is the RAG-specific one and the reason the log exists:

```sql
SELECT c.ticket_id, c.category
FROM corpus_tickets c
LEFT JOIN retrievals r ON c.ticket_id = r.ticket_id
WHERE r.ticket_id IS NULL      -- indexed, never retrieved
```

A ticket nothing ever matches is either irrelevant to what people ask, or written
in language nobody uses. The second is a fixable retrieval problem disguised as a
content problem, and no trace will ever show it — a trace only knows about the
documents that *did* come back.

`LEFT JOIN … IS NULL` rather than `NOT IN`, because `x NOT IN (1, NULL)`
evaluates to NULL rather than true and the query silently returns nothing.

Two things the log refuses to invent:

- **Tokens are NULL when the provider does not report them.** Zero would average
  into every cost report as a measurement. `usage` returns an `unpriced_calls`
  count alongside the total, because a cost figure means nothing without knowing
  how many calls it could not cover.
- **Unknown models have no cost.** Rates live in a table in `analytics.py`; a
  model absent from it produces NULL, not a plausible guess. Self-hosted models
  return 0.0 — that is a real answer, not a missing one.

`DATABASE_URL` is SQLite by default and is the variable most deploy platforms
inject for an attached Postgres, so the log follows the deployment without a code
change. Set `LOG_SEARCHES=false` under load testing, where the log measures the log.

## Evaluation

Three harnesses, each answering a different question:

```bash
python -m evals.retrieval_eval   # did the right ticket come back?  no LLM
python -m evals.simple_eval      # is the answer well-formed and cited?  no judge
python -m evals.ragas_collect    # then score in the eval env (see below)
```

**`retrieval_eval`** is the one to run first and most often. In a RAG pipeline a
wrong answer has two possible causes with opposite fixes, and metrics computed
over the final answer cannot tell them apart. This one never looks at the answer:
14 labelled queries, scored on hit rate, MRR and precision@k, in milliseconds and
for free. Run it after touching the corpus, the embedding model, `k`, or the
score threshold.

**`simple_eval`** is the regression gate for prompt changes. Deterministic checks
on answer shape and citation integrity — including that every `HIST-nnn` in the
answer was actually retrieved, which catches a model inventing a plausible ticket
ID. It cannot tell whether the *claims* are supported; a citation is not a
guarantee of grounding.

**RAGAS** is the only one that measures hallucination, and it runs in two steps
because RAGAS and this application cannot share a virtualenv — RAGAS imports a
`langchain-community` module that the LangChain 1.x line removed:

```bash
python -m evals.ragas_collect                     # app env: run pipeline → JSON

python -m venv .venv-evals
.venv-evals/Scripts/pip install -r requirements-evals.txt
.venv-evals/Scripts/python -m evals.ragas_eval    # eval env: score the JSON
```

The samples file is a durable artefact — you can rescore it with different
metrics, or diff two prompt versions, without paying for another set of model
calls. Results and interpretation live in [RAGAS_RESULTS.md](RAGAS_RESULTS.md).

Judge strength decides what the numbers are worth:

| Provider | Judge | Scores mean |
|---|---|---|
| `bedrock` | Claude 3 Haiku (`JUDGE_MODEL_ID`) | Trustworthy — use as the quality gate |
| `ollama` | Local model (`JUDGE_OLLAMA_MODEL`) | Directional only; a small model follows the rubric loosely and breaks the JSON contract often enough to produce NaNs |
| `fake` | — | Refused. Scoring a canned answer wastes judge tokens on a measurement of nothing |

## Tests

```bash
python -m pytest
```

62 tests, no credentials, no network, no local model. They run against a real
Chroma index and a real SQLite database, both built into temporary directories,
rather than mocked doubles. Mocking retrieval in a RAG project tests the wiring
and skips the part that decides answer quality; mocking the session in the
analytics tests would skip the window functions and the outer join, which are the
only things in that module worth testing.

## Note

A reference implementation built to explore production RAG patterns in support
automation. Contains no proprietary code or data — every ticket in `app/corpus.py`
is synthetic.
