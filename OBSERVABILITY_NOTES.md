# Observability Notes — PTK

What instrumenting this pipeline actually revealed, and what to look for next.

The file is split deliberately. The first half is measured and reproducible on
any machine with one command. The second half needs a real provider and a
LangSmith key, and is left as questions rather than answers — writing plausible
findings there would defeat the point of having observability at all.

---

## Part 1 — Measured offline

Reproduce with:

```bash
LLM_PROVIDER=fake python -m app.ingest --rebuild
LLM_PROVIDER=fake python -m evals.retrieval_eval
```

Configuration: keyword embeddings (`LLM_PROVIDER=fake`), 14 tickets, k=3, no
score threshold.

| Metric | Value | Reading |
|---|---|---|
| Hit rate | 1.000 | Every one of 14 labelled queries retrieved a relevant ticket in the top 3 |
| MRR | 0.929 | The correct ticket is first for 12 of 14 queries, second for 2 |
| precision@3 | 0.357 | Against a labelling ceiling of 0.381 — see below |

### Finding 1 — precision@k is capped by the labels, not the retriever

Twelve of the fourteen cases have exactly one relevant ticket. With k=3 the best
attainable precision on those is 0.33, so the ceiling across the set is 0.381 and
the measured 0.357 is within 0.024 of perfect.

This matters because RAGAS reports context precision on the same 0–1 scale with
no ceiling stated. Reading 0.36 as "the retriever is 36% accurate" would send you
optimising a component that is already at its limit. The fix for a low precision
number is usually a smaller k, not a better retriever — and the eval prints the
ceiling next to the score for exactly this reason.

### Finding 2 — the two misordered results are semantically defensible

| Query | Ranked first | Expected |
|---|---|---|
| "SFTP connection times out before authentication after partner maintenance" | HIST-006 (files not collected) | HIST-005 (firewall/IP change) |
| "Users keep getting signed out, sessions drop after a few minutes" | HIST-011 (JWT secret rotation) | HIST-012 (Redis eviction) |

Both correct tickets came back at rank 2, and in both cases the ticket that
outranked it is one a support engineer would also want to see. This is the
argument for MRR over hit rate: hit rate scores both of these identically to a
perfect result, and MRR does not — and rank matters here because position is the
only signal of importance the model gets.

### Finding 3 — a single similarity threshold cannot separate signal from noise

Distances for the nearest three documents (Chroma returns distance, so lower is
closer):

| Query | 1st | 2nd | 3rd |
|---|---|---|---|
| "sessions drop after a few minutes" (in domain) | 0.265 | 0.344 | 0.558 |
| "SFTP times out before authentication" (in domain) | 0.391 | 0.428 | 0.496 |
| "seating plan for the office move" (out of domain) | **0.444** | 0.580 | 0.677 |

The best match for a completely irrelevant query (0.444) scores better than the
third legitimate result for a relevant one (0.496). Any global cutoff that
rejected the office-move query would also discard real context.

This is why `RETRIEVAL_SCORE_THRESHOLD` exists but is unset by default, and why
the system prompt licenses refusal explicitly. Retrieval cannot represent "we
have nothing on this" — the model has to.

**Caveat that applies to all three findings:** these are keyword embeddings, not
a trained model. The ranking behaviour is real and reproducible; the absolute
distances are not comparable to Titan's. Re-run this section against
`LLM_PROVIDER=bedrock` and replace the numbers.

---

## Part 2 — To fill in from LangSmith

Not yet run. Requires a real provider and `LANGSMITH_API_KEY`:

```bash
python -m app.ingest --rebuild        # with LLM_PROVIDER=bedrock or ollama
python run_ptk_traces.py              # generates 5 traces
```

Then open each trace at https://smith.langchain.com and answer these. Replace
each question with what you actually saw — a specific finding from a real trace
is the difference between describing observability and having used it.

### Latency split

Open any trace and read the durations off the tree:

```
PTK-TICK-5001                         [total: ____ms]
├── RunnableParallel                  [____ms]
│   └── VectorStoreRetriever          [____ms]   ← includes the embedding call
├── format-retrieved-tickets          [____ms]
└── RunnableSequence                  [____ms]
    └── ChatBedrock                   [____ms]
```

- What share of total latency is retrieval vs generation?
- Is the embedding call a meaningful fraction of retrieval? If it is, an
  embedding cache for repeated queries is the highest-value optimisation
  available — and you will have the number to justify it.

### Context budget

Open the `ChatBedrock` node and read the input token count.

- How many tokens do three retrieved tickets consume?
- What fraction of the total prompt is retrieved context vs system prompt?
- At what k would context crowd out the answer? (`BEDROCK_MAX_TOKENS` is 768.)

### Retrieval quality on the paraphrased query

`TICK-5004` is worded the way a customer would word it, not an engineer —
"nothing is coming through from our supplier since Tuesday". Open its retriever
span.

- Did it retrieve HIST-006 (files present but never collected)?
- If not, that is the gap between keyword matching and semantic matching showing
  up on a real query, and it is a corpus or chunking problem rather than a
  prompt one.

### The out-of-domain trace

`TICK-5005` asks about an office seating plan. Retrieval will return three
tickets regardless.

- Did the model decline, or did it construct an answer from irrelevant tickets?
- If it declined: that is the grounding instruction earning its place, and it is
  worth quoting the exact refusal.
- If it did not: tighten the refusal clause in `SYSTEM_PROMPT` and re-run. Record
  both the before and after — a fix you can show the trace for is a stronger
  story than a fix you can only describe.

### Cross-checking against the evals

- Do the tickets cited in the answers match what the retriever span shows? The
  `no_fabricated_ticket_ids` check in `simple_eval` should make this impossible,
  so a mismatch means the check has a hole.
- Filter by `priority:P1` — is latency or answer length systematically different
  from P3? There is no reason it should be; the chain does not branch on priority.

---

## What to write down once Part 2 is filled in

One paragraph per finding, in this shape:

> I opened the trace for [query] and saw [specific observation with a number].
> That told me [which component is responsible]. I changed [the specific lever]
> and the [metric] moved from [before] to [after].

Two or three of those are worth more in an interview than a description of what
LangSmith does — anyone can install it; the finding proves you read the output.
