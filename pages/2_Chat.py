"""Chat page: st.chat_input/st.chat_message loop calling run_query(), with
an optional "search this document" scope. See Phase 4 in the
implementation plan, and Phase 3b for target_document / Phase 3c for the
guardrails that shape what run_query can return."""

import streamlit as st

from rag_learn import config, ratings
from rag_learn.graph import build_graph, build_initial_state
from rag_learn.sync import list_indexed_documents

st.set_page_config(page_title="Chat - RAG_Learn", page_icon="💬")
st.title("💬 Chat")

missing_keys = config.missing_required_keys()
if missing_keys:
    st.error(f"Missing required environment variable(s): {', '.join(missing_keys)}. Set them in `.env`.")
    st.stop()


@st.cache_resource
def get_compiled_graph():
    # Loads the embedding pipeline, reranker, and Groq clients on first
    # call (graph.py's own _Clients lazy singletons) -- st.cache_resource
    # ensures build_graph() itself only runs once per server process, not
    # once per script rerun (Streamlit reruns this whole file on every
    # chat message).
    return build_graph()


compiled_graph = get_compiled_graph()

indexed = list_indexed_documents()
document_options = ["Search everything"] + sorted({name for names in indexed.values() for name in names})
choice = st.sidebar.selectbox("Scope", document_options, help="Restrict retrieval to one known document instead of the whole corpus.")
target_document = None if choice == "Search everything" else choice

if not any(indexed.values()):
    st.info("No documents indexed yet. Go to the **Ingest** page (sidebar) to sync `data/` first.")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources):
            if i > 0:
                st.divider()
            st.markdown(f"**{s.get('source', '?')}**  ·  page {s.get('page', -1)}  ·  `{s.get('file_type', '?')}`")
            if s.get("content"):
                st.text(s["content"][:1500])
            for img_path in s.get("images", []):
                try:
                    st.image(img_path)
                except Exception:
                    st.caption(f"(image not available: {img_path})")


def _render_rating(exchange_id: str) -> None:
    """A 1-5 star widget under each assistant answer, persisted via
    ratings.py (not just Streamlit session state -- see that module's
    docstring for why). Answering "does this look complete and correct to
    the person who asked it" is exactly the judgment a real user rating
    captures that a synthetic eval question can't. A rating of 3 or below
    additionally prompts for what was missing/wrong, since a bare low score
    isn't actionable on its own for the golden-set promotion pipeline
    (eval/promote_candidates.py)."""
    selected = st.feedback("stars", key=f"rating_{exchange_id}")
    if selected is not None:
        rating = selected + 1  # st.feedback is 0-indexed (0 = 1 star)
        comment = None
        if rating <= 3:
            comment = st.text_input(
                "What was missing or wrong?",
                key=f"rating_comment_{exchange_id}",
                label_visibility="collapsed",
                placeholder="What was missing or wrong? (optional)",
            )
        ratings.record_rating(exchange_id, rating, comment or None)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        _render_sources(message.get("sources", []))
        if message["role"] == "assistant" and message.get("exchange_id"):
            _render_rating(message["exchange_id"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = compiled_graph.invoke(build_initial_state(question, target_document))
        st.write(result["generation"])
        _render_sources(result["sources"])
        exchange_id = ratings.record_exchange(question, result["generation"], result["sources"], target_document)
        _render_rating(exchange_id)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["generation"],
            "sources": result["sources"],
            "exchange_id": exchange_id,
        }
    )
