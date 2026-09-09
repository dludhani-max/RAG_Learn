"""Streamlit entry point: `uv run streamlit run streamlit_app.py`.

This file is the Home/status page. Ingest and Chat live in pages/ -- a
standard Streamlit multipage app, so this stays the dashboard rather than
mixing concerns. See Phase 4 in the implementation plan for the design
rationale.
"""

from pathlib import Path

import streamlit as st

from rag_learn import config, vectorless_pageindex, vectorless_sql
from rag_learn.sync import list_indexed_documents
from rag_learn.vectorstore import VectorStore

st.set_page_config(page_title="RAG_Learn", page_icon="📚", layout="centered")

st.title("📚 RAG_Learn")
st.caption(
    "Multi-format agentic RAG over your own documents -- vector, SQL, and page-index "
    "retrieval, orchestrated by LangGraph, with guardrails on both ends."
)

st.subheader("Status")

missing_keys = config.missing_required_keys()
if missing_keys:
    st.error(f"Missing required environment variable(s): {', '.join(missing_keys)}. Set them in `.env` before using Chat.")
else:
    st.success("All required API keys configured.")

col1, col2, col3 = st.columns(3)

with col1:
    try:
        vector_count = VectorStore().count()
    except Exception as e:
        vector_count = None
        st.caption(f"Vector store not available: {e}")
    st.metric("Vector chunks", vector_count if vector_count is not None else "N/A")

with col2:
    try:
        sql_tables = vectorless_sql.list_tables()
    except Exception:
        sql_tables = []
    st.metric("SQL tables", len(sql_tables))

with col3:
    try:
        pageindex_trees = vectorless_pageindex.list_trees()
    except Exception:
        pageindex_trees = []
    st.metric("Page-index trees", len(pageindex_trees))

st.subheader("Indexed documents")
indexed = list_indexed_documents()
if not any(indexed.values()):
    st.info("No documents indexed yet. Go to the **Ingest** page (sidebar) to sync `data/`.")
else:
    labels = {
        "vector": "Vector search",
        "vectorless_sql": "SQL tables",
        "vectorless_pageindex": "Page-index trees",
    }
    for routing, names in indexed.items():
        if names:
            st.markdown(f"**{labels.get(routing, routing)}** ({len(names)}): " + ", ".join(sorted(names)))

st.subheader("Data directory")
st.code(config.DATA_DIR)

st.divider()
st.caption("Use the sidebar to open **Ingest** (sync documents into the store) or **Chat** (ask questions).")
