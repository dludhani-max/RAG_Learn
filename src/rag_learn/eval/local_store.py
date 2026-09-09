"""Local storage for golden-dataset eval results -- always written,
independent of whether LangSmith is reachable. LangSmith's own trace/eval
logging has repeatedly been unusable this session (monthly quota exhausted),
which made every eval run's results unrecoverable once that happened. This
is the durable fallback -- see run_eval.py, which writes here on every run
and only *additionally* pushes to LangSmith on a best-effort basis.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rag_learn import config

DB_PATH = Path(config.VECTOR_STORE_DIR) / "eval_results.duckdb"

# Same short-lived-connection-per-write pattern as telemetry.py/
# vectorless_sql.py -- DuckDB takes an exclusive lock per connection to a
# file, so writes are serialized within this process and never held open.
_write_lock = threading.Lock()


def _connect(read_only: bool = False):
    import duckdb

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_results (
                ts TIMESTAMP,
                experiment VARCHAR,
                question VARCHAR,
                source VARCHAR,
                answer VARCHAR,
                ground_truth VARCHAR,
                contexts_json VARCHAR,
                retrieval_relevance DOUBLE,
                groundedness DOUBLE,
                answer_correctness DOUBLE,
                answer_relevancy DOUBLE,
                langsmith_run_id VARCHAR
            )
            """
        )
    return conn


def record_result(
    experiment: str,
    question: str,
    source: Optional[str],
    answer: str,
    ground_truth: str,
    contexts: list[str],
    scores: dict[str, float],
    langsmith_run_id: Optional[str] = None,
) -> None:
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """INSERT INTO eval_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    datetime.now(),
                    experiment,
                    question,
                    source,
                    answer,
                    ground_truth,
                    json.dumps(contexts),
                    scores.get("retrieval_relevance"),
                    scores.get("groundedness"),
                    scores.get("answer_correctness"),
                    scores.get("answer_relevancy"),
                    langsmith_run_id,
                ],
            )
        finally:
            conn.close()


def list_experiments() -> list[dict]:
    """One row per experiment run: when it ran, how many questions, and the
    average of each metric -- for the Insights page's local eval history."""
    if not DB_PATH.exists():
        return []
    conn = _connect(read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT experiment,
                   MIN(ts) AS started,
                   COUNT(*) AS questions,
                   AVG(retrieval_relevance) AS retrieval_relevance,
                   AVG(groundedness) AS groundedness,
                   AVG(answer_correctness) AS answer_correctness,
                   AVG(answer_relevancy) AS answer_relevancy,
                   SUM(CASE WHEN langsmith_run_id IS NOT NULL THEN 1 ELSE 0 END) AS synced_to_langsmith
            FROM eval_results
            GROUP BY experiment
            ORDER BY started DESC
            """
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def results_for(experiment: str) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _connect(read_only=True)
    try:
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE experiment = ? ORDER BY ts", [experiment]
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


__all__ = ["record_result", "list_experiments", "results_for"]
