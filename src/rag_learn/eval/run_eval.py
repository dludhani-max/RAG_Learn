"""Runs the golden dataset through the full graph and scores each example
on the 4 metrics. Every result is written to the local store
(local_store.py) unconditionally; a push to LangSmith (as a run + feedback
per metric) is attempted best-effort on top of that -- LangSmith's own
trace/eval quota has repeatedly been exhausted this session, which made
earlier runs' results unrecoverable once that happened mid-run. The local
store is now the source of truth; LangSmith is a bonus if reachable, not a
requirement, and a per-example LangSmith failure never drops that
example's local result or stops the run.

Usage: uv run python3 -m rag_learn.eval.run_eval
The golden dataset is read straight from the local reviewed draft file
(golden_dataset_draft.json) -- no LangSmith round-trip is needed just to
get the question set, unlike the original design.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from rag_learn import config
from rag_learn.eval import local_store, metrics
from rag_learn.eval.golden_dataset import DRAFT_PATH
from rag_learn.graph import run_query


def _load_entries() -> list[dict]:
    if not DRAFT_PATH.exists():
        raise FileNotFoundError(
            f"No golden dataset draft at {DRAFT_PATH} -- run golden_dataset.generate_candidates() "
            "and review it first."
        )
    return json.loads(DRAFT_PATH.read_text())


def _get_langsmith_client():
    """None if LangSmith isn't configured or unreachable -- callers must
    treat that as "skip LangSmith for this run," not an error."""
    try:
        from langsmith import Client

        client = Client()
        # list_runs() is lazy (a generator) -- must actually be consumed to
        # trigger the request and find out whether LangSmith is reachable.
        next(iter(client.list_runs(project_name=config.LANGSMITH_PROJECT, limit=1)), None)
        return client
    except Exception as e:
        print(f"[EVAL] LangSmith unavailable this run, logging locally only: {e}")
        return None


def _push_to_langsmith(client, experiment: str, question: str, answer: str, scores: Dict[str, float]) -> Optional[str]:
    """Best-effort: log this example as a standalone run + one feedback
    entry per metric. Returns the LangSmith run_id on success, None on any
    failure -- the caller must not let this raise, a quota 429 here should
    never cost the local result."""
    run_id = None
    try:
        import uuid

        run_id = str(uuid.uuid4())
        client.create_run(
            id=run_id,
            name=experiment,
            run_type="chain",
            inputs={"question": question},
            outputs={"answer": answer},
            project_name=config.LANGSMITH_PROJECT,
        )
        client.update_run(run_id, end_time=datetime.now())
        for key, score in scores.items():
            client.create_feedback(run_id=run_id, key=key, score=score)
        return run_id
    except Exception as e:
        print(f"[EVAL] LangSmith push failed for this example (kept locally): {e}")
        return None


def run_eval(experiment: Optional[str] = None) -> str:
    """Runs every golden example through the real graph, scores it, and
    records it. Returns the experiment label used (auto-generated from a
    timestamp if not given), so callers can look it up afterward via
    local_store.results_for()."""
    experiment = experiment or f"rag-learn-eval-{datetime.now():%Y%m%d-%H%M%S}"
    entries = _load_entries()
    client = _get_langsmith_client()

    for i, entry in enumerate(entries):
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        source = entry.get("source")
        print(f"[{i + 1}/{len(entries)}] {question}")

        result = run_query(question, target_document=source)
        answer = result["generation"]
        contexts = [s.get("content", "") for s in result.get("sources", [])]
        scores = {
            "retrieval_relevance": metrics.retrieval_relevance(question, contexts),
            "groundedness": metrics.groundedness(answer, contexts),
            "answer_correctness": metrics.answer_correctness(answer, ground_truth),
            "answer_relevancy": metrics.answer_relevancy(question, answer),
        }
        print(f"  scores: {scores}")

        langsmith_run_id = _push_to_langsmith(client, experiment, question, answer, scores) if client else None
        local_store.record_result(
            experiment=experiment,
            question=question,
            source=source,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            scores=scores,
            langsmith_run_id=langsmith_run_id,
        )

    return experiment


def _print_summary(experiment: str) -> None:
    rows = local_store.results_for(experiment)
    keys = ["retrieval_relevance", "groundedness", "answer_correctness", "answer_relevancy"]
    print(f"\n=== {experiment} ({len(rows)} questions) ===")
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            print(f"{k}: avg={sum(vals) / len(vals):.3f}")
    synced = sum(1 for r in rows if r.get("langsmith_run_id"))
    print(f"Synced to LangSmith: {synced}/{len(rows)}")


if __name__ == "__main__":
    exp = run_eval()
    _print_summary(exp)
