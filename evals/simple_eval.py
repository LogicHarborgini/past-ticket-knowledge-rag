"""
Local answer eval for PTK — no LangSmith account or judge model required.

Run from the project root, after `python -m app.ingest`:

    python -m evals.simple_eval

Every criterion is a deterministic string or shape check, so this costs nothing
beyond the pipeline invocations themselves and never disagrees with itself
between runs. It is the regression gate: change the system prompt, rerun, compare
against evals/baseline_results.json.

What it can and cannot see: it can tell that an answer cited HIST-009 and that
HIST-009 was retrieved. It cannot tell whether the *claims* in the answer are
supported by HIST-009 — a citation is not a guarantee of grounding, and no
keyword check can close that gap. That is what evals/ragas_eval.py is for.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.providers import active_model_id, resolve_provider
from app.ptk_chain import ainvoke_ptk_traced
from app.vectorstore import document_count

BASELINE_PATH = Path(__file__).parent / "baseline_results.json"

# Words the prompt licenses the model to use when the retrieved tickets do not
# apply. Matching on intent phrases rather than an exact sentence keeps the
# check from being a test of one wording.
DECLINE_PHRASES = [
    "do not address", "does not address", "don't address", "doesn't address",
    "not relevant", "no relevant", "not related", "unrelated",
    "do not appear", "does not appear", "cannot help", "no historical",
    "nothing in the", "not covered",
]


@dataclass(frozen=True)
class EvalCriterion:
    """
    One pass/fail check.

    check       (answer, retrieved_ids, case) -> bool
    applies_to  (case) -> bool. Criteria that only make sense for some queries
                are skipped rather than counted as failures — declines_politely
                would otherwise punish the model for answering a question it
                could answer.
    needs_model whether the criterion judges model output. Under
                LLM_PROVIDER=fake the answer is a fixed string, so these are
                skipped: scoring them would measure the canned text.
    """

    name: str
    check: Callable[[str, list[str], dict], bool]
    applies_to: Callable[[dict], bool] = lambda case: True
    needs_model: bool = True


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


CRITERIA: list[EvalCriterion] = [
    EvalCriterion(
        name="retrieved_something",
        check=lambda _answer, retrieved, case: bool(retrieved),
        # An empty index or a broken embedding model fails here first, before
        # any answer-shaped criterion can produce a misleading partial score.
        needs_model=False,
    ),
    EvalCriterion(
        name="cites_a_retrieved_ticket",
        # The citation must name a ticket that was actually retrieved. Checking
        # only for "some HIST-nnn" would pass a model that invented a plausible
        # ID, which is the exact failure citations are supposed to prevent.
        check=lambda answer, retrieved, _case: any(tid in answer for tid in retrieved),
        applies_to=lambda case: case["expect"] == "answer",
    ),
    EvalCriterion(
        name="no_fabricated_ticket_ids",
        check=lambda answer, retrieved, _case: all(
            tid in retrieved for tid in re.findall(r"HIST-\d{3}", answer)
        ),
    ),
    EvalCriterion(
        name="mentions_expected_ticket",
        # The strongest available offline signal: for these queries there is one
        # ticket that plainly holds the answer, and it should be the one cited.
        check=lambda answer, _retrieved, case: case["expected_ticket"] in answer,
        applies_to=lambda case: case["expect"] == "answer",
    ),
    EvalCriterion(
        name="declines_when_irrelevant",
        check=lambda answer, _retrieved, _case: _contains_any(answer, DECLINE_PHRASES),
        applies_to=lambda case: case["expect"] == "decline",
    ),
    EvalCriterion(
        name="not_too_short",
        check=lambda answer, _retrieved, _case: len(answer.split()) >= 20,
        applies_to=lambda case: case["expect"] == "answer",
    ),
    EvalCriterion(
        # System prompt says under 150 words. 200 leaves room for citations
        # without letting an essay through.
        name="respects_length_guidance",
        check=lambda answer, _retrieved, _case: len(answer.split()) <= 200,
    ),
]


TEST_CASES = [
    {
        "id": "EVAL-001",
        "priority": "P1",
        "expect": "answer",
        "expected_ticket": "HIST-009",
        "query": (
            "Our connection pool is exhausted on production and the nightly batch "
            "will not start. Logs show max_connections exceeded."
        ),
    },
    {
        "id": "EVAL-002",
        "priority": "P1",
        "expect": "answer",
        "expected_ticket": "HIST-003",
        "query": (
            "We are sending AS2 messages successfully but MDN receipts never come "
            "back from the partner."
        ),
    },
    {
        "id": "EVAL-003",
        "priority": "P2",
        "expect": "answer",
        "expected_ticket": "HIST-007",
        "query": (
            "The partner's API keeps returning 429 Too Many Requests even though "
            "we lowered our polling frequency."
        ),
    },
    {
        "id": "EVAL-004",
        "priority": "P3",
        "expect": "decline",
        "expected_ticket": "",
        # Out of domain on purpose. Retrieval will still return its three nearest
        # tickets — that is what a k-nearest search does — so this measures
        # whether the model declines the ones it was handed. A pipeline that
        # answers this confidently is the one that will invent a resolution for a
        # real customer.
        "query": (
            "What is the recommended seating plan for the office move next month "
            "and who approves the budget?"
        ),
    },
]


@dataclass
class EvalResult:
    case_id: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def score(self) -> float:
        applicable = len(self.passed) + len(self.failed)
        return len(self.passed) / applicable if applicable else 0.0


async def run_evals() -> dict:
    if document_count() == 0:
        raise SystemExit(
            "Vector index is empty. Run `python -m app.ingest` before evaluating."
        )

    provider = resolve_provider()
    real_model = provider != "fake"

    print(f"provider={provider} | model={active_model_id()}")
    if not real_model:
        print(
            "LLM_PROVIDER=fake — the answer is a fixed string, so answer-quality\n"
            "criteria are skipped. This run proves the harness works, not the model."
        )
    print()

    results: list[EvalResult] = []

    for case in TEST_CASES:
        result = EvalResult(case_id=case["id"])

        try:
            payload, _run_id = await ainvoke_ptk_traced(
                query=case["query"],
                ticket_id=case["id"],
                priority=case["priority"],
            )
        except Exception as e:
            # A failed invocation scores 0 rather than aborting the run — a
            # partial baseline is still worth writing.
            result.error = str(e)
            results.append(result)
            print(f"{case['id']} ({case['expect']}) — ERROR: {e}")
            continue

        answer = payload["answer"]
        retrieved = [source["ticket_id"] for source in payload["sources"]]
        result.retrieved = retrieved

        for criterion in CRITERIA:
            if not criterion.applies_to(case) or (criterion.needs_model and not real_model):
                result.skipped.append(criterion.name)
            elif criterion.check(answer, retrieved, case):
                result.passed.append(criterion.name)
            else:
                result.failed.append(criterion.name)

        results.append(result)

        print(f"{case['id']} ({case['expect']}) — score: {result.score:.0%} "
              f"| retrieved: {', '.join(retrieved) or 'none'}")
        if result.passed:
            print(f"  pass: {', '.join(result.passed)}")
        if result.failed:
            print(f"  FAIL: {', '.join(result.failed)}")
        if result.skipped:
            print(f"  n/a:  {', '.join(result.skipped)}")

    overall = sum(r.score for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 55)
    print(f"Overall: {overall:.0%} across {len(results)} case(s)")

    summary = {
        "provider": provider,
        "model": active_model_id(),
        # Written into the artefact, not just printed. A committed
        # "overall_score": 1.0 with no context reads as a perfect model; it is
        # only a perfect harness when the model was fake.
        "note": (
            None if real_model else
            "provider=fake: every criterion that judges model output was skipped. "
            "This score measures the harness, not answer quality."
        ),
        "overall_score": round(overall, 4),
        "cases": len(results),
        "results": [
            {
                "case_id": r.case_id,
                "score": round(r.score, 4),
                "retrieved": r.retrieved,
                "passed": r.passed,
                "failed": r.failed,
                "skipped": r.skipped,
                "error": r.error,
            }
            for r in results
        ],
    }

    _compare_to_baseline(summary)
    BASELINE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Baseline written to {BASELINE_PATH.name}")

    return summary


def _compare_to_baseline(summary: dict) -> None:
    """Report movement against the previous run, if there is one."""
    if not BASELINE_PATH.exists():
        print("No previous baseline — this run establishes it.")
        return

    try:
        previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read previous baseline ({e}) — overwriting.")
        return

    if previous.get("provider") != summary["provider"]:
        # A fake-provider baseline and a Bedrock one skip different criteria, so
        # their overall scores are not the same measurement.
        print(
            f"Baseline was recorded on provider '{previous.get('provider')}', "
            f"this run used '{summary['provider']}' — not comparable."
        )
        return

    delta = summary["overall_score"] - previous.get("overall_score", 0.0)
    if abs(delta) < 0.001:
        print(f"No change vs baseline ({previous['overall_score']:.0%}).")
    elif delta > 0:
        print(f"IMPROVED: {previous['overall_score']:.0%} -> {summary['overall_score']:.0%} (+{delta:.0%})")
    else:
        print(f"REGRESSED: {previous['overall_score']:.0%} -> {summary['overall_score']:.0%} ({delta:.0%})")


if __name__ == "__main__":
    asyncio.run(run_evals())
