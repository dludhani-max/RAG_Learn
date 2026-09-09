"""Ingest-only CLI: sync data/ into whichever retrieval mechanism each file
routes to (vector, SQL, or page-index), no query, no LLM generation calls.
For a full pipeline smoke test (ingest + one query through the graph), use
`uv run rag-learn` instead. For interactive use, the Streamlit UI.

Uses sync.py rather than a blind full re-ingest, so re-running this is cheap
-- only new/modified/removed files under DATA_DIR are processed."""

from rag_learn.sync import sync

if __name__ == "__main__":
    summary = sync()
    print(f"\nDone. {summary}")
