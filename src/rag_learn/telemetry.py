"""Local token/cost telemetry for every LLM call across both providers
(Groq primary, Anthropic fallback -- see llm_factory.py). Captured via a
LangChain callback attached at the one central construction point, so no
individual call site needs to know telemetry exists. Persisted to a local
DuckDB table (same short-lived-connection-per-write pattern as
vectorless_sql.py, since DuckDB takes an exclusive lock per connection) so
usage is visible across process restarts, not just the current session.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from rag_learn import config

DB_PATH = Path(config.VECTOR_STORE_DIR) / "telemetry.duckdb"

# $ per 1M (input, output) tokens -- point-in-time estimates for display
# only, not a billing source of truth; update if provider pricing changes.
# Groq's on-demand tier is currently free/quota-based rather than metered,
# so its cost is reported as 0.0 -- token counts and how often the
# Anthropic fallback fires are the meaningful signal there, not dollars.
_PRICING: dict[str, tuple[float, float]] = {
    "groq": (0.0, 0.0),
    "anthropic": (1.00, 5.00),  # Claude Haiku 4.5 list pricing as of writing
    "openrouter": (0.0, 0.0),  # the ":free" models in config.OPENROUTER_MODEL_NAMES
}

# Serializes writes within this process -- vectorless_pageindex.py's
# concurrent section summarization (a bounded thread pool) can trigger
# several of these at once, and DuckDB's per-connection exclusive file lock
# means overlapping connections to the same file fail (same constraint
# vectorless_sql.py already works around with short-lived connections).
_write_lock = threading.Lock()


def _connect(read_only: bool = False):
    import duckdb

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                ts TIMESTAMP,
                provider VARCHAR,
                model VARCHAR,
                purpose VARCHAR,
                is_fallback BOOLEAN,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd DOUBLE
            )
            """
        )
    return conn


def _record(
    provider: str, model: str, purpose: str, is_fallback: bool, input_tokens: int, output_tokens: int
) -> None:
    in_price, out_price = _PRICING.get(provider, (0.0, 0.0))
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [datetime.now(), provider, model, purpose, is_fallback, input_tokens, output_tokens, cost],
            )
        finally:
            conn.close()


def _provider_for(class_name: str, model: str) -> str:
    """Fallback heuristic when no "provider:<name>" tag is present (e.g. a
    manually-constructed client bypassing llm_factory.py, as in the
    session's own smoke tests). Unreliable for OpenRouter specifically,
    since it's constructed via the same ChatOpenAI class real OpenAI would
    use -- the factory always tags its clients explicitly so this path isn't
    normally hit in production use."""
    haystack = f"{class_name} {model}".lower()
    if "anthropic" in haystack or "claude" in haystack:
        return "anthropic"
    if "groq" in haystack:
        return "groq"
    return "unknown"


def _provider_from_tags(tags) -> Optional[str]:
    for t in tags or []:
        if isinstance(t, str) and t.startswith("provider:"):
            return t.split(":", 1)[1]
    return None


class TelemetryCallback(BaseCallbackHandler):
    """Attached once per LLMFactory-built client. Records every actual LLM
    invocation -- primary or fallback, whichever one really executed -- to
    the local telemetry store. A `.with_fallbacks()` runnable only invokes
    one branch per call, so exactly one on_chat_model_start/on_llm_end pair
    fires per real request; there's no double-counting between primary and
    fallback attempts."""

    def __init__(self, purpose: str, primary_provider: str = "groq"):
        self.purpose = purpose
        self.primary_provider = primary_provider
        self._starts: dict[str, tuple[str, str]] = {}

    def on_chat_model_start(self, serialized: dict, messages, *, run_id, **kwargs: Any) -> None:
        serialized = serialized or {}
        raw_id = serialized.get("id")
        class_name = raw_id[-1] if isinstance(raw_id, list) and raw_id else serialized.get("name", "")
        # The model name lives in the serialized constructor kwargs (verified
        # live: kwargs["invocation_params"] does NOT carry it for either
        # ChatGroq or ChatAnthropic, despite the name suggesting it would).
        ctor_kwargs = serialized.get("kwargs") or {}
        model = ctor_kwargs.get("model") or ctor_kwargs.get("model_name") or ""
        # llm_factory tags every client it builds with "provider:<name>" --
        # authoritative when present, since it's the only reliable signal
        # for OpenRouter (constructed via the same ChatOpenAI class real
        # OpenAI would use, so class-name/model-string sniffing can't tell
        # them apart). Falls back to the heuristic only for a client built
        # outside llm_factory.
        provider = _provider_from_tags(kwargs.get("tags")) or _provider_for(class_name, model)
        self._starts[str(run_id)] = (provider, model)

    def on_llm_end(self, response, *, run_id, **kwargs: Any) -> None:
        provider, model = self._starts.pop(str(run_id), (self.primary_provider, ""))
        input_tokens = output_tokens = 0
        try:
            message = response.generations[0][0].message
            usage = getattr(message, "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
            elif response.llm_output:
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        except Exception as e:
            print(f"[TELEMETRY] Could not read token usage: {e}")
        is_fallback = provider != self.primary_provider
        try:
            _record(provider, model, self.purpose, is_fallback, input_tokens, output_tokens)
        except Exception as e:
            # Telemetry must never break a real LLM call -- fail silently
            # (loud print, no raise), matching this codebase's fail-open
            # philosophy for every non-essential side path.
            print(f"[TELEMETRY] Failed to record usage: {e}")


def summary() -> list[dict]:
    """Aggregated usage for the Insights page: totals per provider/purpose,
    plus how often the Anthropic fallback actually fired."""
    if not DB_PATH.exists():
        return []
    conn = _connect(read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT provider, purpose, is_fallback,
                   COUNT(*) AS calls,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(cost_usd) AS cost_usd
            FROM llm_calls
            GROUP BY provider, purpose, is_fallback
            ORDER BY provider, purpose
            """
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def recent(limit: int = 200) -> list[dict]:
    """The most recent individual calls, newest first -- for a detail view
    alongside the aggregated summary() above."""
    if not DB_PATH.exists():
        return []
    conn = _connect(read_only=True)
    try:
        rows = conn.execute("SELECT * FROM llm_calls ORDER BY ts DESC LIMIT ?", [limit]).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


__all__ = ["TelemetryCallback", "summary", "recent"]
