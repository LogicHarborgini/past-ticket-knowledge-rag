"""
Evaluation harnesses for PTK.

Run these as modules from the project root so that `app` is importable:

    python -m evals.simple_eval     # deterministic checks, no judge model
    python -m evals.retrieval_eval  # retrieval-only: hit rate, MRR, precision@k
    python -m evals.ragas_eval      # RAGAS: faithfulness, relevance, precision

Running them as file paths (python evals/simple_eval.py) puts evals/ on sys.path
instead of the project root, and `import app` then fails.

Three harnesses rather than one because they answer different questions and cost
different amounts:

    retrieval_eval  Did the right ticket come back? No LLM at all, runs in
                    milliseconds, and it is the first thing to check when
                    answers are wrong — a generation metric cannot distinguish
                    "the model ignored good context" from "the context was junk".
    simple_eval     Is the answer well-formed and does it cite its sources?
                    Deterministic, free, the regression gate for prompt changes.
    ragas_eval      Is the answer actually grounded in what was retrieved?
                    Needs a judge LLM, takes minutes, and is the only one of the
                    three that measures hallucination.
"""
