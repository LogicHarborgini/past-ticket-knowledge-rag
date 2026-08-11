"""
The historical ticket corpus this pipeline retrieves from.

Every ticket here is synthetic. The production system indexes real customer
tickets; nothing proprietary is reproduced, and the scenarios are the generic
failure modes of any B2B integration platform — EDI translation, AS2 receipts,
SFTP transport, API rate limits, database connection pools.

Two decisions in the shape of this data matter more than they look:

1. `to_document_text()` puts the resolution *inside* the embedded text rather
   than parking it in metadata. It costs a little retrieval precision — the
   resolution wording dilutes the issue wording the query is matched against —
   and buys something worth more: whatever the model is asked to ground its
   answer in is exactly what an evaluator sees as the context. With resolutions
   hidden in metadata, RAGAS scores faithfulness against text that never
   contained the answer, and a correct pipeline reads as a hallucinating one.

2. Ticket IDs are stable and used as Chroma document IDs, so re-running
   ingestion upserts instead of accumulating duplicate copies of the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalTicket:
    """One resolved ticket: what broke, and what fixed it."""

    ticket_id: str
    partner: str
    category: str
    issue: str
    resolution: str

    def to_document_text(self) -> str:
        """The text that gets embedded and shown to the model as context."""
        return f"Issue: {self.issue}\nResolution: {self.resolution}"

    def to_metadata(self) -> dict[str, str]:
        """
        Filterable fields kept alongside the vector.

        Chroma metadata values must be scalars (str/int/float/bool) — no lists,
        no nested dicts.
        """
        return {
            "ticket_id": self.ticket_id,
            "partner": self.partner,
            "category": self.category,
        }


HISTORICAL_TICKETS: list[HistoricalTicket] = [
    HistoricalTicket(
        ticket_id="HIST-001",
        partner="Acme Corp",
        category="EDI",
        issue=(
            "EDI 850 purchase orders stopped processing overnight. 47 documents "
            "queued and the translator rejects every one with error X12-834."
        ),
        resolution=(
            "The partner upgraded from X12 version 4010 to 5010 without notice, so "
            "the inbound map no longer matched the envelope. Updated the trading "
            "partner profile to 5010, regenerated the 850 map, restarted the EDI "
            "translator service and replayed the 47 queued documents from the "
            "error queue. All processed on the second attempt."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-002",
        partner="Globex",
        category="EDI",
        issue=(
            "Outbound 810 invoices are being generated with empty line-item "
            "segments. Header data is correct, detail loop is missing."
        ),
        resolution=(
            "A source query change dropped the join to the order-detail table, so "
            "the map received a header row with no children. Restored the join, "
            "added a pre-transmission validation rule that fails the document when "
            "the detail loop count is zero rather than sending an empty invoice."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-003",
        partner="Northwind Logistics",
        category="AS2",
        issue=(
            "AS2 messages send successfully but no MDN receipts come back. "
            "Transmissions have been running unacknowledged since 06:00."
        ),
        resolution=(
            "The partner's MDN callback URL pointed at an endpoint whose TLS "
            "certificate had expired two days earlier, so their server refused the "
            "receipt handshake. Renewed the certificate, confirmed the chain was "
            "complete including the intermediate, and requested asynchronous MDN "
            "resend for the affected message IDs."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-004",
        partner="Initech",
        category="AS2",
        issue=(
            "Inbound AS2 messages are rejected with a decryption failure. Partner "
            "insists nothing changed on their side."
        ),
        resolution=(
            "The partner had rotated their signing certificate and published the "
            "new public key without sending it. Imported the current certificate "
            "into the trading partner profile and added a calendar reminder 30 days "
            "before its expiry so the next rotation is coordinated, not discovered."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-005",
        partner="DataSync",
        category="SFTP",
        issue=(
            "SFTP connection failures started after the partner's maintenance "
            "window. Connection times out before authentication."
        ),
        resolution=(
            "The partner moved to a new load balancer with a different egress IP "
            "range and our firewall whitelist still held the old one. Timing out "
            "before the auth prompt is the signature of a network block rather "
            "than a credential problem. Updated the whitelist with the new range "
            "and verified with a manual SSH handshake on port 22."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-006",
        partner="Umbrella Health",
        category="SFTP",
        issue=(
            "Files land on the SFTP server but the pickup job never collects them. "
            "The directory listing shows the files present."
        ),
        resolution=(
            "The partner's process wrote files directly to the pickup directory, so "
            "the poller skipped anything still being written and never revisited "
            "it. Switched the partner to write-then-rename into a staging path and "
            "configured the poller to collect only completed filenames."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-007",
        partner="TechFlow",
        category="API",
        issue=(
            "REST API calls started returning 429 Too Many Requests. Reducing call "
            "frequency did not clear it."
        ),
        resolution=(
            "Three integration jobs shared one API key, so the per-key rate limit "
            "was being consumed collectively even though each job individually "
            "looked well under it. Issued separate keys per job and added "
            "exponential backoff with jitter to the client so a throttle spreads "
            "retries instead of synchronising them into a second burst."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-008",
        partner="Soylent Retail",
        category="API",
        issue=(
            "Webhook deliveries to the partner endpoint fail intermittently with "
            "504 Gateway Timeout. Roughly one delivery in five."
        ),
        resolution=(
            "The partner processed each webhook synchronously, and payloads above "
            "about 2 MB exceeded their 30-second gateway timeout. They moved to "
            "acknowledging on receipt and processing on a queue. We reduced the "
            "batch size and enabled delivery retries with backoff."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-009",
        partner="Acme Corp",
        category="Database",
        issue=(
            "Database connection pool exhausted. Logs full of max_connections "
            "exceeded and the end-of-day batch cannot start."
        ),
        resolution=(
            "The reporting service opened connections inside a loop and never "
            "returned them, so the leak only surfaced under end-of-day volume. "
            "Wrapped the connections in context managers so they close on the "
            "error path too, and raised the pool ceiling from 10 to 50 as "
            "headroom rather than as the fix."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-010",
        partner="Globex",
        category="Database",
        issue=(
            "Queries that used to return instantly now take eight seconds and "
            "time out downstream. Started right after the data migration."
        ),
        resolution=(
            "The migration recreated the table without its index on customer_id, "
            "so every lookup became a full scan. Rebuilt the index and added an "
            "index-presence check to the migration runbook's post-step "
            "verification. Query time went from about 8s back to 12ms."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-011",
        partner="Initech",
        category="Authentication",
        issue=(
            "All users receive 401 Unauthorized on the integration portal. "
            "Credentials are confirmed correct."
        ),
        resolution=(
            "The JWT signing secret was rotated in the secrets manager but the "
            "running service still held the previous value in memory, so every "
            "token it issued failed validation. Restarted the service to pick up "
            "the new secret and added secret rotation to the deployment checklist."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-012",
        partner="Umbrella Health",
        category="Authentication",
        issue=(
            "Sessions drop after a few minutes and users are forced to sign in "
            "repeatedly through the afternoon."
        ),
        resolution=(
            "The Redis session store hit its memory ceiling with maxmemory-policy "
            "set to noeviction, so new sessions were refused and existing ones "
            "were lost on write. Switched the policy to allkeys-lru, raised the "
            "memory limit and added an alert at 80 percent utilisation."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-013",
        partner="Northwind Logistics",
        category="Deployment",
        issue=(
            "Release rolled back automatically. The migration step failed during "
            "deployment and the pipeline reverted."
        ),
        resolution=(
            "The migration used a syntax accepted by PostgreSQL 13 but removed in "
            "14, and staging was still on 13 so it passed there. Corrected the "
            "statement and aligned the staging database version with production so "
            "the next version-specific break is caught before release."
        ),
    ),
    HistoricalTicket(
        ticket_id="HIST-014",
        partner="TechFlow",
        category="Deployment",
        issue=(
            "Scheduled mapping job reports success but writes an output file "
            "containing only headers. Source shows 1,200 records available."
        ),
        resolution=(
            "The job ran against a stale connection profile pointing at an empty "
            "sandbox schema, so it genuinely had nothing to write and correctly "
            "reported success. Repointed the profile at production and made the "
            "job fail rather than succeed when the record count is zero."
        ),
    ),
]


def corpus_documents() -> tuple[list[str], list[dict[str, str]], list[str]]:
    """
    The corpus in the three parallel lists Chroma's add_texts() expects.

    Returns
    -------
    tuple[list[str], list[dict[str, str]], list[str]]
        (texts, metadatas, ids) — ids are the ticket IDs, which makes ingestion
        an upsert rather than an append.
    """
    texts = [t.to_document_text() for t in HISTORICAL_TICKETS]
    metadatas = [t.to_metadata() for t in HISTORICAL_TICKETS]
    ids = [t.ticket_id for t in HISTORICAL_TICKETS]
    return texts, metadatas, ids
