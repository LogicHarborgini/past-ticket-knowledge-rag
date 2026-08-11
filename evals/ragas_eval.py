"""
Step 2 of RAGAS evaluation: score the collected samples.

Run from the project root, in the *eval* environment (see requirements-evals.txt
for why it is a separate virtualenv):

    .venv-evals/Scripts/python -m evals.ragas_eval        # Windows
    .venv-evals/bin/python -m evals.ragas_eval            # macOS / Linux

Reads evals/ragas_samples.json, writes evals/ragas_baseline.json.

This module deliberately imports nothing from `app`. It runs in an environment
pinned to RAGAS's LangChain requirements, where the application's own imports
would fail — which is exactly why the pipeline run and the scoring run are two
scripts with a JSON file between them.

The four metrics, and what a low one is telling you:

    faithfulness        Is every claim in the answer supported by the retrieved
                        context? Low means hallucination — the model is
                        supplementing the tickets with its own ideas. Fix in the
                        prompt.
    answer_relevancy    Does the answer address the question? Low means the
                        answer is grounded but off-target — it summarised the
                        retrieved tickets instead of applying them.
    context_precision   Were the retrieved documents relevant? Low means the
                        retriever is padding the context with noise. Fix in
                        retrieval: k, chunking, threshold, embedding model.
    context_recall      Did retrieval find everything the reference answer
                        needed? Low means the right ticket never came back at
                        all, and no prompt change will help.

The diagnostic that matters is the pair. Low faithfulness with high context
precision is a generation problem; low on both is a retrieval problem wearing a
generation problem's clothes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).parent
SAMPLES_PATH = EVALS_DIR / "ragas_samples.json"
BASELINE_PATH = EVALS_DIR / "ragas_baseline.json"

# A drop larger than this between runs is called a regression rather than noise.
# LLM-as-judge scoring is not deterministic — the same samples rescored will move
# by a couple of points — so a threshold below about 0.03 would fire on the
# judge's own variance.
REGRESSION_THRESHOLD = 0.05

# RAGAS column name -> the name used in reports and the baseline file.
METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


# ─────────────────────────────────────────────────────────────────────────────
# Judge model
# ─────────────────────────────────────────────────────────────────────────────


def _has_aws() -> bool:
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def build_judge():
    """
    The LLM and embeddings RAGAS uses to score.

    This repeats a little of app/providers.py rather than importing it, because
    this file runs in an environment where `app` is not installable. The
    duplication is the cost of the split, and it is small and deliberate.

    Judge choice changes what the numbers are worth. RAGAS asks the judge to
    decompose an answer into claims and rule on each one against the context; a
    3B local model follows that rubric loosely and breaks the JSON contract often
    enough to produce NaNs. Treat Ollama scores as directional and Bedrock or
    OpenAI scores as reportable.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # JUDGE_PROVIDER overrides, falling back to LLM_PROVIDER. The override exists
    # because "openai" is a valid judge and not a valid setting for the
    # application — the app would treat it as unknown and fall back to auto,
    # leaving the two halves silently disagreeing about what is running.
    provider = os.getenv("JUDGE_PROVIDER", os.getenv("LLM_PROVIDER", "auto")).strip().lower()
    if provider in {"auto", "fake"}:
        # A fake app provider says nothing about which judge is available; probe.
        provider = "bedrock" if _has_aws() else "ollama"

    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings, ChatBedrock

        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        judge = ChatBedrock(
            model_id=os.getenv("JUDGE_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            region_name=region,
            # Temperature 0: a judge that scores the same answer differently on
            # consecutive runs turns every regression check into a coin toss.
            model_kwargs={"temperature": 0, "max_tokens": 1024},
        )
        embeddings = BedrockEmbeddings(
            model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            region_name=region,
        )
        label = f"bedrock:{os.getenv('JUDGE_MODEL_ID', 'claude-3-haiku')}"

    elif provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        judge = ChatOpenAI(model=os.getenv("JUDGE_OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        embeddings = OpenAIEmbeddings()
        label = f"openai:{os.getenv('JUDGE_OPENAI_MODEL', 'gpt-4o-mini')}"

    else:
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        model = os.getenv("JUDGE_OLLAMA_MODEL", "llama3.2")
        judge = ChatOllama(model=model, temperature=0)
        embeddings = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        )
        label = f"ollama:{model}"
        print(
            "Judge is a local model — scores are directional only. Expect some "
            "metrics to come back NaN when it breaks the JSON contract.\n"
        )

    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(embeddings), label


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def load_samples() -> dict[str, Any]:
    if not SAMPLES_PATH.exists():
        raise SystemExit(
            f"{SAMPLES_PATH.name} not found. Run the collection step first, in the "
            "application environment:\n    python -m evals.ragas_collect"
        )
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


def score(collected: dict[str, Any]) -> tuple[dict[str, float | None], list[dict], str]:
    """
    Run RAGAS over the collected samples.

    Returns
    -------
    tuple[dict, list[dict], str]
        Mean score per metric (None where every sample failed to score), the
        per-sample rows, and the judge label. The per-sample rows are the useful
        half: a mean of 0.7 across six questions can be six mediocre answers or
        five good ones and a disaster, and only the second is actionable.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    judge_llm, judge_embeddings, judge_label = build_judge()
    print(f"Judge: {judge_label}")

    dataset = EvaluationDataset.from_list([
        {
            "user_input": sample["user_input"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "response": sample["response"],
            "reference": sample["reference"],
        }
        for sample in collected["samples"]
    ])

    print(f"Scoring {len(dataset)} samples — this takes a few minutes.\n")

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        # raise_exceptions=False: a judge that fails on one sample leaves a NaN
        # in that cell instead of destroying the run. Five scored samples and one
        # hole is a usable result; a traceback is not.
        raise_exceptions=False,
    )

    frame = result.to_pandas()

    means: dict[str, float | None] = {}
    for column, name in METRIC_COLUMNS.items():
        if column not in frame.columns:
            means[name] = None
            continue
        series = frame[column].dropna()
        means[name] = round(float(series.mean()), 3) if len(series) else None

    rows = []
    for index, sample in enumerate(collected["samples"]):
        row = {"question": sample["user_input"][:80]}
        for column, name in METRIC_COLUMNS.items():
            value = frame[column].iloc[index] if column in frame.columns else None
            row[name] = None if value is None or value != value else round(float(value), 3)
        rows.append(row)

    return means, rows, judge_label


def interpret(means: dict[str, float | None]) -> list[str]:
    """
    Plain-English reading of the scores — the answer to "so what do you do about
    it?", which is the follow-up question every interviewer asks after the
    numbers.
    """
    lines: list[str] = []
    faith = means.get("faithfulness")
    precision = means.get("context_precision")
    relevance = means.get("answer_relevancy")
    recall = means.get("context_recall")

    if faith is not None and precision is not None:
        if faith < 0.7 and precision < 0.7:
            lines.append(
                "Both faithfulness and context precision are low: retrieval is "
                "feeding the model poor context and the model is filling the gap "
                "itself. Fix retrieval first — prompt changes cannot ground an "
                "answer in documents that were never retrieved."
            )
        elif faith < 0.7:
            lines.append(
                "Faithfulness is low while context precision holds up: the right "
                "tickets are being retrieved and the model is adding claims they "
                "do not support. This is a prompt problem — tighten the grounding "
                "instruction in SYSTEM_PROMPT."
            )
        elif precision < 0.7:
            lines.append(
                "Context precision is low while faithfulness holds up: the model "
                "is ignoring the noise it is handed, which works until it does "
                "not. Lower RETRIEVAL_TOP_K or set RETRIEVAL_SCORE_THRESHOLD."
            )
        else:
            lines.append(
                "Faithfulness and context precision are both healthy — answers "
                "are grounded in relevant retrieved tickets."
            )

    if relevance is not None and relevance < 0.7:
        lines.append(
            "Answer relevancy is low: responses are drifting from the question "
            "asked. Instruct the prompt to lead with the action that resolves the "
            "stated issue before adding context."
        )

    if recall is not None and recall < 0.7:
        lines.append(
            "Context recall is low: the reference answers need information that "
            "retrieval never returned. Raise k or check the corpus actually "
            "covers these questions — this one cannot be fixed downstream."
        )

    scored = {name: value for name, value in means.items() if value is not None}
    if scored:
        weakest = min(scored, key=scored.get)
        lines.append(f"Weakest metric: {weakest} ({scored[weakest]:.3f}) — start there.")

    return lines


def compare_to_baseline(means: dict[str, float | None], previous: dict) -> bool:
    """Print movement against the stored baseline. Returns True on a regression."""
    old_scores = previous.get("scores", {})
    print("\nvs baseline (" + previous.get("scored_at", "unknown date")[:10] + "):")

    regressed = False
    for name, value in means.items():
        old = old_scores.get(name)
        if old is None or value is None:
            print(f"  {name:<18} {str(old):>6} -> {str(value):>6}   (not comparable)")
            continue
        delta = value - old
        flag = "  REGRESSION" if delta < -REGRESSION_THRESHOLD else ""
        regressed = regressed or bool(flag)
        print(f"  {name:<18} {old:.3f} -> {value:.3f}   {delta:+.3f}{flag}")

    return regressed


def main() -> int:
    collected = load_samples()

    if collected.get("provider") == "fake":
        raise SystemExit(
            "The samples were collected with LLM_PROVIDER=fake — every response "
            "is the same canned string. Recollect against a real provider before "
            "scoring; there is nothing here worth a judge's tokens."
        )

    print(f"Samples:    {len(collected['samples'])} (collected {collected['collected_at'][:19]})")
    print(f"Generation: {collected['generation_model']}")
    print(f"Embeddings: {collected['embedding_model']} | k={collected['top_k']}")

    means, rows, judge_label = score(collected)

    print("\n" + "=" * 68)
    print("RAGAS RESULTS")
    print("=" * 68)
    for name, value in means.items():
        display = f"{value:.3f}" if value is not None else "  n/a  (all samples failed to score)"
        print(f"  {name:<18} {display}")

    print("\nPer question:")
    for row in rows:
        scores = "  ".join(
            f"{name[:4]}={row[name]:.2f}" if row[name] is not None else f"{name[:4]}=n/a"
            for name in METRIC_COLUMNS.values()
        )
        print(f"  {scores}   {row['question']}")

    reading = interpret(means)
    print("\nReading:")
    for line in reading:
        print(f"  - {line}")

    output = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "collected_at": collected["collected_at"],
        "judge": judge_label,
        "generation_model": collected["generation_model"],
        "embedding_model": collected["embedding_model"],
        "top_k": collected["top_k"],
        "samples": len(collected["samples"]),
        "scores": means,
        "per_question": rows,
        "interpretation": reading,
    }

    regressed = False
    if BASELINE_PATH.exists():
        try:
            previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"\nCould not read previous baseline ({e}) — overwriting.")
        else:
            if previous.get("judge") != judge_label:
                # Two judges disagree with each other more than a prompt change
                # moves either of them, so a cross-judge delta is not a signal.
                print(
                    f"\nBaseline was judged by '{previous.get('judge')}', this run "
                    f"by '{judge_label}' — scores are not comparable."
                )
            else:
                regressed = compare_to_baseline(means, previous)

    BASELINE_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nBaseline written to {BASELINE_PATH.name}")
    print("Copy the scores into RAGAS_RESULTS.md — and nowhere else until they are real.")

    # Non-zero on regression so this can gate a CI job later.
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
