"""Vectorless SQL retrieval path: tabular files (routing == "vectorless_sql")
bypass embeddings entirely and are queried exactly, via text-to-SQL against
an in-memory-shaped DuckDB table, rather than lossily chunked and searched
by similarity.

See Phase 3b.i in the implementation plan for the full design/rationale.
"""

import contextlib
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import duckdb

from rag_learn import config
from rag_learn.classifier import ROUTING_SQL

DB_PATH = Path(config.VECTOR_STORE_DIR) / "vectorless.duckdb"
SOURCES_PATH = Path(config.VECTOR_STORE_DIR) / "vectorless_sql_sources.json"

# Guardrail: this executes LLM-generated SQL against real data, so only a
# single read-only SELECT (optionally with a leading WITH/CTE) is ever
# allowed. Reject anything containing a mutating/DDL/session keyword
# outright, even if it also happens to start with SELECT (e.g. a
# stacked-statement injection attempt via a semicolon).
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|"
    r"EXPORT|IMPORT|CALL|VACUUM|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
_READ_ONLY_START_RE = re.compile(r"^\s*(WITH\b.*?\bSELECT\b|SELECT)\b", re.IGNORECASE | re.DOTALL)


@contextlib.contextmanager
def _connect() -> Iterator[duckdb.DuckDBPyConnection]:
    """Short-lived connection per call, not a persistent module-level
    singleton. DuckDB takes an exclusive lock on the database file for the
    life of a connection -- verified live: a persistent connection here
    left the file locked for a running Streamlit server's entire process
    lifetime, so a separate `sync()` CLI run against the same data
    directory failed outright with a DuckDB IOException the moment it
    touched a SQL-routed table. Opening and closing per call keeps the
    lock held only for one ingest/query's duration, so a long-running UI
    session and an occasional CLI invocation can coexist as long as they
    don't land on the exact same instant."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    try:
        yield conn
    finally:
        conn.close()


def _load_sources() -> Dict[str, str]:
    if not SOURCES_PATH.exists():
        return {}
    import json

    try:
        return json.loads(SOURCES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sources(sources: Dict[str, str]) -> None:
    import json

    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(json.dumps(sources, indent=2))


def table_name(path: Path) -> str:
    """Deterministic DuckDB table name from a source filename -- distinct
    from _stable_id-style hashing (vectorstore.py) since a human-readable
    name here also documents itself directly in generated SQL/citations."""
    stem = re.sub(r"[^a-z0-9_]", "_", path.stem.lower()).strip("_") or "table"
    if stem[0].isdigit():
        stem = f"t_{stem}"
    return stem


def ingest_table(path: Path) -> str:
    """Load a tabular file into its own DuckDB table, replacing any prior
    version (idempotent re-ingestion, matching sync.py's modified-file
    flow). Returns the table name."""
    table = table_name(path)
    with _connect() as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')

        suffix = path.suffix.lower()
        if suffix == ".csv":
            conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM read_csv_auto(?)', [str(path)])
        elif suffix in (".xlsx", ".xls"):
            import pandas as pd

            df = pd.read_excel(path)
            conn.register("_vectorless_sql_tmp", df)
            conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _vectorless_sql_tmp')
            conn.unregister("_vectorless_sql_tmp")
        else:
            raise ValueError(f"Unsupported tabular file type for SQL routing: {suffix}")

        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    sources = _load_sources()
    sources[table] = str(path)
    _save_sources(sources)
    print(f"[SQL] Ingested {path.name} -> table '{table}' ({row_count} rows)")
    return table


def drop_table(table: str) -> None:
    """Remove a table -- used by sync.py when a vectorless_sql-routed file
    is modified (drop-then-reingest) or deleted."""
    with _connect() as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    sources = _load_sources()
    sources.pop(table, None)
    _save_sources(sources)


def list_tables() -> List[str]:
    with _connect() as conn:
        return [row[0] for row in conn.execute("SHOW TABLES").fetchall()]


def _table_schema(conn: duckdb.DuckDBPyConnection, table: str) -> str:
    rows = conn.execute(f'DESCRIBE "{table}"').fetchall()
    return ", ".join(f"{r[0]} ({r[1]})" for r in rows)


def _is_read_only_select(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if not stripped or _FORBIDDEN_SQL_RE.search(stripped):
        return False
    return bool(_READ_ONLY_START_RE.match(stripped))


_sql_llm = None


def _get_sql_llm():
    global _sql_llm
    if _sql_llm is None:
        _sql_llm = config.get_llm(temperature=0.0, purpose="sql_generation")
    return _sql_llm


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()


def _generate_sql(question: str, table: str, schema: str) -> str:
    prompt = (
        "You are a DuckDB SQL generator. Given the table schema below, write exactly one "
        "read-only SELECT query that answers the question. Output only the raw SQL -- no "
        "explanation, no markdown code fences, no comments. Only reference the given table.\n\n"
        "Select every column relevant to the question, not just the one literally asked for -- "
        "e.g. for 'which engineers earn over 100k', select name, department, AND salary, not just "
        "name, so the result table is self-explanatory to someone who can't see this SQL. Only "
        "narrow to fewer columns for a genuine aggregate (COUNT/SUM/AVG/etc.).\n\n"
        f"Table: {table}\n"
        f"Schema: {schema}\n\n"
        f"Question: {question}\n\nSQL:"
    )
    try:
        raw = _get_sql_llm().invoke(prompt).content
    except Exception as e:
        print(f"[ERROR] SQL generation LLM call failed for table '{table}': {e}")
        return ""
    return _strip_code_fence(raw)


def _rows_to_markdown(columns: List[str], rows: List[tuple]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend("| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def query(question: str, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Text-to-SQL retrieval: generate and run a read-only SELECT per
    candidate table, returning results shaped like the vector path's
    retrieved documents so graph.py's generate() node can consume either
    uniformly."""
    sources = _load_sources()
    results: List[Dict[str, Any]] = []

    with _connect() as conn:
        candidate_tables = tables if tables is not None else [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        for table in candidate_tables:
            schema = _table_schema(conn, table)
            sql = _generate_sql(question, table, schema)
            if not sql:
                continue
            if not _is_read_only_select(sql):
                print(f"[SQL] Rejected non-read-only SQL for table '{table}': {sql!r}")
                continue
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
            except Exception as e:
                print(f"[ERROR] SQL execution failed on table '{table}' ({sql!r}): {e}")
                continue
            if not rows:
                continue
            results.append(
                {
                    "content": _rows_to_markdown(columns, rows),
                    "metadata": {
                        "source_file": Path(sources.get(table, table)).name,
                        "file_type": "table",
                        "page": -1,
                        "sql": sql,
                    },
                    "score": None,
                }
            )
    return results


__all__ = ["ROUTING_SQL", "ingest_table", "drop_table", "list_tables", "query", "table_name"]
