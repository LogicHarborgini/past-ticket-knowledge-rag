"""
Generate sample PTK traces to populate the LangSmith dashboard.

Run from the project root, after `python -m app.ingest`:

    python run_ptk_traces.py

Each query produces one trace named PTK-<ticket_id>, tagged with its priority,
with the retriever and format-retrieved-tickets as child spans alongside the
model call. Requires a working provider — AWS credentials for Bedrock, a running
Ollama daemon, or LLM_PROVIDER=fake to exercise the plumbing with no model.
"""

from __future__ import annotations

import asyncio

from app.ptk_chain import ainvoke_ptk_traced
from app.vectorstore import document_count

# Deliberately spread across categories, priorities and query styles. A
# dashboard where every trace is the same shape tells you nothing about how
# retrieval behaves at the edges — so the last two are the interesting ones:
#
#   TICK-5004 uses vocabulary the corpus does not (a customer's words, not an
#   engineer's), which is where embeddings earn their keep over keyword search.
#   TICK-5005 is out of domain entirely. The retriever still returns its three
#   nearest neighbours; the answer is supposed to decline anyway. Open that
#   trace first — the gap between what was retrieved and what was answered is
#   the whole argument for tracing retrieval separately from generation.
SAMPLE_QUERIES = [
    {
        "ticket_id": "TICK-5001",
        "priority": "P1",
        "query": (
            "Production database connection pool is exhausted. max_connections "
            "exceeded in the logs and the end-of-day batch cannot start."
        ),
    },
    {
        "ticket_id": "TICK-5002",
        "priority": "P1",
        "query": (
            "AS2 messages are sending but no MDN receipts are coming back. "
            "Partner says they are receiving the files."
        ),
    },
    {
        "ticket_id": "TICK-5003",
        "priority": "P2",
        "query": (
            "EDI 850 documents are failing translation with error X12-834 after "
            "the partner's weekend maintenance."
        ),
    },
    {
        "ticket_id": "TICK-5004",
        "priority": "P2",
        "query": (
            "Nothing is coming through from our supplier since Tuesday. The "
            "files show up on the server but our system never picks them up."
        ),
    },
    {
        "ticket_id": "TICK-5005",
        "priority": "P3",
        "query": (
            "What is the recommended seating plan for the office move next month?"
        ),
    },
]


async def main() -> None:
    indexed = document_count()
    if indexed == 0:
        print("Vector index is empty. Run `python -m app.ingest` first.")
        return

    print(f"Index holds {indexed} tickets\n")

    for item in SAMPLE_QUERIES:
        ticket_id = item["ticket_id"]
        # ASCII arrow deliberately. A Windows console defaults to cp1252, which
        # has no U+2192, and print() raises UnicodeEncodeError rather than
        # degrading — a script that dies on its own progress output.
        print(f"-> {ticket_id} ({item['priority']}): {item['query'][:70]}...")

        try:
            result, run_id = await ainvoke_ptk_traced(
                query=item["query"],
                ticket_id=ticket_id,
                priority=item["priority"],
            )
        except Exception as e:
            # Keep going: one bad query should not cost you the whole batch.
            print(f"  failed: {e}\n")
            continue

        cited = ", ".join(s["ticket_id"] for s in result["sources"]) or "none"
        print(f"  retrieved: {cited}")
        print(f"  answer:    {result['answer'][:160].strip()}...")
        print(f"  run_id:    {run_id}\n")

    print("Done — check https://smith.langchain.com under your LANGSMITH_PROJECT")


if __name__ == "__main__":
    asyncio.run(main())
