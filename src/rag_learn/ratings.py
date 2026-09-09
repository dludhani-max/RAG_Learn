"""Persists every Chat answer (question, answer, sources) so it can be
rated -- Streamlit's session_state alone doesn't survive a refresh or a new
session, and there was nowhere durable to attach a rating before this.
Ratings feed the golden-dataset promotion pipeline (see
eval/promote_candidates.py): a highly-rated real exchange is a strong
candidate for the golden set, since it reflects an actual user judging
whether they got a complete, correct answer -- not a synthetic question.

Same short-lived-connection-per-write DuckDB pattern as telemetry.py/
eval/local_store.py.
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rag_learn import config

DB_PATH = Path(config.VECTOR_STORE_DIR) / "ratings.duckdb"

_write_lock = threading.Lock()


def _connect(read_only: bool = False):
    import duckdb

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exchanges (
                id VARCHAR PRIMARY KEY,
                ts TIMESTAMP,
                question VARCHAR,
                answer VARCHAR,
                sources_json VARCHAR,
                target_document VARCHAR,
                rating INTEGER,
                rating_comment VARCHAR,
                rated_ts TIMESTAMP,
                promoted BOOLEAN DEFAULT FALSE
            )
            """
        )
    return conn


def record_exchange(question: str, answer: str, sources: list[dict], target_document: Optional[str]) -> str:
    """Called once per Chat answer, right after generation -- returns an id
    the UI keeps (in st.session_state) so a later rating can be attached to
    this specific exchange."""
    exchange_id = str(uuid.uuid4())
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """INSERT INTO exchanges
                   (id, ts, question, answer, sources_json, target_document, rating, rating_comment, rated_ts, promoted)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, FALSE)""",
                [exchange_id, datetime.now(), question, answer, json.dumps(sources), target_document],
            )
        finally:
            conn.close()
    return exchange_id


def record_rating(exchange_id: str, rating: int, comment: Optional[str] = None) -> None:
    """rating is 1-5 (the UI is responsible for converting st.feedback's
    0-indexed return value)."""
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                "UPDATE exchanges SET rating = ?, rating_comment = ?, rated_ts = ? WHERE id = ?",
                [rating, comment, datetime.now(), exchange_id],
            )
        finally:
            conn.close()


def get_rating(exchange_id: str) -> Optional[dict]:
    conn = _connect(read_only=True)
    try:
        row = conn.execute(
            "SELECT rating, rating_comment FROM exchanges WHERE id = ?", [exchange_id]
        ).fetchone()
        if row is None:
            return None
        return {"rating": row[0], "rating_comment": row[1]}
    finally:
        conn.close()


def list_promotable(min_rating: int = 4) -> list[dict]:
    """Rated at/above min_rating, not yet promoted to a golden-dataset
    candidate -- what eval/promote_candidates.py works through."""
    if not DB_PATH.exists():
        return []
    conn = _connect(read_only=True)
    try:
        rows = conn.execute(
            "SELECT * FROM exchanges WHERE rating >= ? AND promoted = FALSE ORDER BY ts", [min_rating]
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def mark_promoted(exchange_id: str) -> None:
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("UPDATE exchanges SET promoted = TRUE WHERE id = ?", [exchange_id])
        finally:
            conn.close()


def stats() -> dict[str, Any]:
    """For the Insights page: how many exchanges exist, how many are rated,
    the rating distribution."""
    if not DB_PATH.exists():
        return {"total": 0, "rated": 0, "avg_rating": None, "promotable": 0}
    conn = _connect(read_only=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
        rated_row = conn.execute(
            "SELECT COUNT(*), AVG(rating) FROM exchanges WHERE rating IS NOT NULL"
        ).fetchone()
        promotable = conn.execute(
            "SELECT COUNT(*) FROM exchanges WHERE rating >= 4 AND promoted = FALSE"
        ).fetchone()[0]
        return {
            "total": total,
            "rated": rated_row[0],
            "avg_rating": round(rated_row[1], 2) if rated_row[1] is not None else None,
            "promotable": promotable,
        }
    finally:
        conn.close()


__all__ = ["record_exchange", "record_rating", "get_rating", "list_promotable", "mark_promoted", "stats"]
