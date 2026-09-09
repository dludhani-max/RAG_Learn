def main() -> None:
    """`rag-learn` CLI entry point: a smoke test that the whole pipeline
    works end-to-end -- sync data/ into whichever retrieval mechanisms it
    routes to, then ask one question through the full guarded graph. For
    ingestion only (no query, no LLM calls), use `app.py` instead. For
    interactive use, use the Streamlit UI (`streamlit run streamlit_app.py`)."""
    from rag_learn.graph import run_query
    from rag_learn.sync import sync

    sync()

    result = run_query("What is Deepak's experience?")
    print("\n=== Generation ===")
    print(result["generation"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(f"- {s['source']} (page {s['page']}, {s['file_type']})")
