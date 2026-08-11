"""
Retrieval-only evaluation for PTK — no LLM, no judge, no API key.

Run from the project root, after `python -m app.ingest`:

    python -m evals.retrieval_eval

Why this exists separately from the answer evals: in a RAG pipeline a wrong
answer has two possible causes with opposite fixes — the retriever fetched the
wrong tickets, or it fetched the right ones and the model ignored them. Metrics
computed over the final answer cannot tell those apart. These can, because they
never look at the answer at all.

It is also the only harness that runs in milliseconds and costs nothing, which
makes it the right thing to run after touching the corpus, the embedding model,
k, or the score threshold.

Metrics, all at the configured k:

    hit rate     fraction of queries where at least one relevant ticket was
                 retrieved. The blunt one: below ~0.9 the pipeline is broken and
                 nothing downstream is worth measuring.
    MRR          mean reciprocal rank of the first relevant ticket. Rewards
                 putting the right ticket first, not merely somewhere in k —
                 which matters because the model reads position as importance.
    precision@k  fraction of retrieved tickets that were relevant. This is the
                 one RAGAS calls context precision, computed here from labels
                 instead of an LLM's opinion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.providers import active_embedding_model_id
from app.vectorstore import document_count, get_retriever

logging.disable(logging.WARNING)   # the harness prints its own report

BASELINE_PATH = Path(__file__).parent / "retrieval_baseline.json"


@dataclass(frozen=True)
class RetrievalCase:
    """
    One labelled query.

    `relevant` holds every ticket that genuinely helps with this query, not just
    the single best one. Labelling only the obvious match would punish the
    retriever for finding a second ticket that a support engineer would also want
    — and precision@k would then be capped at 1/k by the labels rather than by
    the retrieval.
    """

    query: str
    relevant: frozenset[str]
    note: str = ""


CASES: list[RetrievalCase] = [
    RetrievalCase(
        query="Database connection pool exhausted, max_connections exceeded, batch job blocked",
        relevant=frozenset({"HIST-009"}),
        note="near-verbatim phrasing — the easy case, and the canary if it fails",
    ),
    RetrievalCase(
        query="Queries got slow after the data migration and now time out downstream",
        relevant=frozenset({"HIST-010"}),
    ),
    RetrievalCase(
        query="AS2 MDN receipts never arrive, transmissions unacknowledged",
        relevant=frozenset({"HIST-003"}),
    ),
    RetrievalCase(
        query="Inbound AS2 decryption failure, partner says nothing changed",
        relevant=frozenset({"HIST-004"}),
    ),
    RetrievalCase(
        query="EDI 850 purchase orders rejected with X12-834 and queued in error",
        relevant=frozenset({"HIST-001"}),
    ),
    RetrievalCase(
        query="Invoices going out with no line items, only header data",
        relevant=frozenset({"HIST-002"}),
    ),
    RetrievalCase(
        query="SFTP connection times out before authentication after partner maintenance",
        relevant=frozenset({"HIST-005"}),
    ),
    RetrievalCase(
        query="Files are on the SFTP server but the pickup job never collects them",
        relevant=frozenset({"HIST-006"}),
    ),
    RetrievalCase(
        query="API calls throttled with 429 even after reducing frequency",
        relevant=frozenset({"HIST-007"}),
    ),
    RetrievalCase(
        query="Webhook deliveries intermittently fail with 504 gateway timeout",
        relevant=frozenset({"HIST-008"}),
    ),
    RetrievalCase(
        query="Everyone is getting 401 unauthorized even with correct credentials",
        relevant=frozenset({"HIST-011"}),
    ),
    RetrievalCase(
        query="Users keep getting signed out, sessions drop after a few minutes",
        relevant=frozenset({"HIST-012"}),
    ),
    RetrievalCase(
        # Two tickets legitimately apply: the migration that broke a release, and
        # the job that succeeded while writing nothing. Both are deployment-time
        # failures a stale-config question should surface.
        query="Deployment rolled back automatically when the migration step failed",
        relevant=frozenset({"HIST-013", "HIST-014"}),
        note="two relevant tickets — checks the labels are not single-answer by habit",
    ),
    RetrievalCase(
        query="Certificate expired and the partner endpoint stopped accepting our calls",
        relevant=frozenset({"HIST-003", "HIST-004"}),
        note="cert failures span both AS2 tickets",
    ),
]


@dataclass
class CaseResult:
    query: str
    retrieved: list[str] = field(default_factory=list)
    relevant: frozenset[str] = frozenset()

    @property
    def hit(self) -> bool:
        return any(ticket in self.relevant for ticket in self.retrieved)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first relevant ticket; 0 when none was retrieved."""
        for position, ticket in enumerate(self.retrieved, start=1):
            if ticket in self.relevant:
                return 1.0 / position
        return 0.0

    @property
    def precision(self) -> float:
        if not self.retrieved:
            return 0.0
        hits = sum(1 for ticket in self.retrieved if ticket in self.relevant)
        return hits / len(self.retrieved)


def run_retrieval_eval() -> dict:
    indexed = document_count()
    if indexed == 0:
        raise SystemExit(
            "Vector index is empty. Run `python -m app.ingest` before evaluating."
        )

    k = settings.retrieval_top_k
    retriever = get_retriever(k)
    results: list[CaseResult] = []

    print(f"Index: {indexed} tickets | embeddings: {active_embedding_model_id()} | k={k}\n")

    for case in CASES:
        documents = retriever.invoke(case.query)
        retrieved = [doc.metadata.get("ticket_id", "UNKNOWN") for doc in documents]
        result = CaseResult(query=case.query, retrieved=retrieved, relevant=case.relevant)
        results.append(result)

        marker = "ok  " if result.hit else "MISS"
        expected = "/".join(sorted(case.relevant))
        print(f"{marker} rr={result.reciprocal_rank:.2f} p@{k}={result.precision:.2f} "
              f"| want {expected:<19} | got {', '.join(retrieved)}")
        print(f"      {case.query[:88]}")

    total = len(results)
    summary = {
        "embedding_model": active_embedding_model_id(),
        "k": k,
        "score_threshold": settings.retrieval_score_threshold,
        "cases": total,
        "hit_rate": round(sum(r.hit for r in results) / total, 4),
        "mrr": round(sum(r.reciprocal_rank for r in results) / total, 4),
        "precision_at_k": round(sum(r.precision for r in results) / total, 4),
        "misses": [r.query for r in results if not r.hit],
    }

    print("\n" + "=" * 60)
    print(f"hit rate     {summary['hit_rate']:.0%}   (at least one relevant ticket in top-{k})")
    print(f"MRR          {summary['mrr']:.3f}  (1.0 = the right ticket is always first)")
    print(f"precision@{k}  {summary['precision_at_k']:.3f}  (share of retrieved tickets that are relevant)")

    # precision@k has a ceiling below 1.0 whenever a case has fewer relevant
    # tickets than k — with one labelled ticket and k=3 the best possible score
    # is 0.33. Stating the ceiling stops the number being read as a failure.
    ceiling = sum(min(len(c.relevant), k) / k for c in CASES) / total
    print(f"             ceiling for these labels at k={k} is {ceiling:.3f}")

    if summary["misses"]:
        print(f"\n{len(summary['misses'])} miss(es) — the retriever found nothing relevant for:")
        for query in summary["misses"]:
            print(f"  - {query}")

    _compare_to_baseline(summary)
    BASELINE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBaseline written to {BASELINE_PATH.name}")

    return summary


def _compare_to_baseline(summary: dict) -> None:
    """Report movement against the previous run, if there is one."""
    if not BASELINE_PATH.exists():
        print("\nNo previous baseline — this run establishes it.")
        return

    try:
        previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"\nCould not read previous baseline ({e}) — overwriting.")
        return

    if previous.get("embedding_model") != summary["embedding_model"]:
        # Scores from two embedding models are not comparable. Reporting a delta
        # between them would invite reading a model swap as a regression.
        print(
            f"\nBaseline was recorded with '{previous.get('embedding_model')}', "
            f"this run used '{summary['embedding_model']}' — not comparable."
        )
        return

    print("\nvs baseline:")
    for metric in ("hit_rate", "mrr", "precision_at_k"):
        old = previous.get(metric)
        if old is None:
            continue
        delta = summary[metric] - old
        direction = "same" if abs(delta) < 0.001 else ("up" if delta > 0 else "DOWN")
        print(f"  {metric:<15} {old:.3f} -> {summary[metric]:.3f}  ({direction} {delta:+.3f})")


if __name__ == "__main__":
    run_retrieval_eval()
