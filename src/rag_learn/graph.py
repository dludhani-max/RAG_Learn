"""LangGraph orchestration: a Corrective/Adaptive RAG flow.

input_safety_guardrail -> check_cache -> retrieve -> grade_documents
    -> [generate | transform_query (retry loop) | no_related_answer]
    -> output_guardrail -> store_cache

See Phase 3 in the implementation plan for the design rationale, and Phase
3c for the guardrail nodes. Phase 3's original web_search (Tavily) fallback
node has been removed: per an explicit decision, the answer must come only
from what's actually indexed (vector/vectorless RAG), never from the open
web, so "local retrieval found nothing relevant" is now a terminal
no-related-answer response instead of a fallback trigger.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from rag_learn import config, guardrails, vectorless_pageindex, vectorless_sql
from rag_learn.cache import QACache
from rag_learn.embedding import EmbeddingPipeline
from rag_learn.reranker import Reranker
from rag_learn.search import Retriever
from rag_learn.vectorstore import VectorStore

MIN_RELEVANT_DOCS = 1

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Safety net: strip any <think>...</think> block that slips through
    even with reasoning_effort='none' (e.g. a future model/version change)."""
    return _THINK_BLOCK_RE.sub("", text).strip()


class GraphState(TypedDict):
    question: str
    original_question: str
    documents: list[dict[str, Any]]
    generation: str
    no_relevant_docs: bool
    retry_count: int
    sources: list[dict[str, Any]]
    cache_hit: bool
    # Optional Phase 3b hook: a filename (matched against a vectorless_sql
    # table, a vectorless_pageindex tree, or a vector-routed source_file)
    # scoping retrieval to one known document -- "search this document" in
    # the plan's option (a), set via the Streamlit Chat page's "Scope"
    # picker. None means "search everything": retrieve() fans out across
    # all three retrieval paths (vector + every SQL table + every
    # page-index tree) and merges results, letting grade_documents filter
    # for relevance -- initially deferred per the plan's own recommendation,
    # but adopted once unscoped queries against vectorless-routed content
    # were verified live to always return "no related answer" without it.
    target_document: Optional[str]
    # Phase 3c guardrails: set by input_safety_guardrail when the question
    # itself must be refused outright (prompt injection/abuse) -- routes
    # straight to END, bypassing cache/retrieval/generation entirely.
    blocked: bool
    # Internal handoff from generate() to output_guardrail's groundedness
    # check -- not meaningful to any other node.
    _last_context: str
    # Set by generate() on an LLM-call failure (e.g. exhausted API quota) --
    # verified live: without this, a transient "service unavailable"
    # message got cached as if it were a real answer, then kept being
    # served for that question indefinitely (the cache only invalidates on
    # document changes, not time), long after the service recovered.
    _skip_cache: bool
    # First-pass vectorless (SQL + page-index) results, cached so a
    # transform_query retry doesn't re-pay for the entire fan-out -- see
    # retrieve()'s retry branch below.
    _vectorless_results: list[dict[str, Any]]
    # time.monotonic() at input_safety_guardrail -- lets _route_after_grading
    # enforce a hard time budget regardless of retry_count (see
    # QUERY_TIME_BUDGET_SECONDS below).
    _query_start_time: float


class _Clients:
    """Lazily-initialized singletons shared across graph invocations in this
    process, so the embedding model and vector store connection are loaded
    once, not per query."""

    _pipeline: Optional[EmbeddingPipeline] = None
    _store: Optional[VectorStore] = None
    _retriever: Optional[Retriever] = None
    _reranker: Optional[Reranker] = None
    _cache: Optional[QACache] = None
    _utility_llm: Optional[Runnable] = None
    _generation_llm: Optional[Runnable] = None

    @classmethod
    def pipeline(cls) -> EmbeddingPipeline:
        if cls._pipeline is None:
            cls._pipeline = EmbeddingPipeline()
        return cls._pipeline

    @classmethod
    def retriever(cls) -> Retriever:
        if cls._retriever is None:
            cls._store = cls._store or VectorStore()
            cls._retriever = Retriever(cls._store, cls.pipeline())
        return cls._retriever

    @classmethod
    def cache(cls) -> QACache:
        if cls._cache is None:
            cls._cache = QACache()
        return cls._cache

    @classmethod
    def reranker(cls) -> Reranker:
        if cls._reranker is None:
            cls._reranker = Reranker()
        return cls._reranker

    @classmethod
    def utility_llm(cls) -> Runnable:
        # Grading/query-rewrite are short structured tasks -- suppressing
        # reasoning output (config.llm_kwargs, model-family-aware) skips
        # burning output tokens on internal deliberation nobody reads for a
        # yes/no or one-line rewrite.
        if cls._utility_llm is None:
            cls._utility_llm = config.get_llm(temperature=0.0, purpose="utility_grading")
        return cls._utility_llm

    @classmethod
    def generation_llm(cls) -> Runnable:
        if cls._generation_llm is None:
            cls._generation_llm = config.get_llm(temperature=0.1, purpose="generation")
        return cls._generation_llm


# --- Nodes -----------------------------------------------------------------


def input_safety_guardrail(state: GraphState) -> dict[str, Any]:
    """Phase 3c: the first thing that runs on every query, before cache
    lookup or retrieval. Two checks:
    1. Safety classification (prompt injection / abuse) -- a flagged
       question never touches the cache or retrieval at all.
    2. PII redaction on the question itself -- deterministic, no extra LLM
       call, so it always runs regardless of the safety verdict.

    Also records _query_start_time here (not in build_initial_state) so it
    reflects actual processing start, used by _route_after_grading's hard
    time budget.

    Corpus-relevance ("is this actually answerable from what's indexed?")
    is NOT checked here -- that's grade_documents' job, downstream, since
    it needs to see actual retrieval results to judge accurately."""
    start_time = time.monotonic()
    redacted_question = guardrails.redact_pii(state["question"])
    is_safe, refusal = guardrails.check_input_safety(redacted_question)
    if not is_safe:
        return {
            "blocked": True,
            "question": redacted_question,
            "original_question": redacted_question,
            "generation": refusal,
            "sources": [],
            "_query_start_time": start_time,
        }
    return {
        "blocked": False,
        "question": redacted_question,
        "original_question": redacted_question,
        "_query_start_time": start_time,
    }


def check_cache(state: GraphState) -> dict[str, Any]:
    """Semantic Q&A cache lookup -- if a close-enough question was already
    answered (by anyone, cache is shared, not per-session), reuse that
    answer instead of running retrieval/grading/generation again."""
    embedding = _Clients.pipeline().model.encode([state["question"]])[0]
    hit = _Clients.cache().lookup(state["question"], embedding)
    if hit is None:
        return {"cache_hit": False}

    print(
        f"[CACHE] Hit (similarity={hit['similarity']:.3f}) for '{state['question']}' "
        f"~= cached '{hit['cached_question']}'"
    )
    return {"cache_hit": True, "generation": hit["generation"], "sources": hit["sources"]}


def store_cache(state: GraphState) -> dict[str, Any]:
    embedding = _Clients.pipeline().model.encode([state["original_question"]])[0]
    _Clients.cache().store(state["original_question"], embedding, state["generation"], state["sources"])
    return {}


def _match_sql_table(target: str) -> Optional[str]:
    table = vectorless_sql.table_name(Path(target))
    return table if table in vectorless_sql.list_tables() else None


def _match_pageindex_tree(target: str) -> Optional[Path]:
    wanted = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(target).stem)
    for tree_path in vectorless_pageindex.list_trees():
        if tree_path.stem == wanted:
            return tree_path
    return None


def retrieve(state: GraphState) -> dict[str, Any]:
    target = state.get("target_document")

    if target:
        # Phase 3b.i/3b.ii: a scoped query goes straight to the matching
        # vectorless path instead of embedding search -- these already
        # return exact rows / a specific section, so there's nothing for
        # the cross-encoder reranker to usefully reorder.
        table = _match_sql_table(target)
        if table:
            return {"documents": vectorless_sql.query(state["question"], tables=[table])}
        tree_path = _match_pageindex_tree(target)
        if tree_path:
            return {"documents": vectorless_pageindex.query(state["question"], tree_paths=[tree_path])}
        # target_document isn't a known SQL table or page-index tree --
        # assume it's a vector-routed source_file and scope vector search
        # to just that document's chunks (falls through below).

    # Retrieve-then-rerank: fetch a wider candidate set by (cheap) vector
    # similarity, then a cross-encoder scores each pair jointly for a more
    # accurate ranking, keeping only the final top_k.
    candidates = _Clients.retriever().retrieve(
        state["question"],
        top_k=config.RETRIEVE_CANDIDATES,
        score_threshold=config.SCORE_THRESHOLD,
        source_file=target,
    )
    docs = _Clients.reranker().rerank(state["question"], candidates, top_k=config.TOP_K)

    if not target:
        # "Search everything" must actually search everything. Vector
        # search alone never sees vectorless_sql/vectorless_pageindex-routed
        # content -- that's the whole point of routing it away from
        # embeddings (Phase 2.5/3b). Verified live: an unscoped question
        # about a page-index-routed document (most of this corpus) always
        # returned "No related answers found" even though the answer
        # existed, because nothing ever checked the other two retrieval
        # paths without an explicit target_document. grade_documents (next
        # node) already filters the combined list for actual relevance, so
        # merging in every vectorless result here and letting grading sort
        # it out is safe, not just additive noise.
        if state["retry_count"] == 0:
            # First pass: run the full vectorless fan-out and cache the
            # result on state for any retry to reuse.
            vectorless_results = vectorless_sql.query(state["question"]) + vectorless_pageindex.query(
                state["question"]
            )
            docs = docs + vectorless_results
            return {"documents": docs, "_vectorless_results": vectorless_results}
        # A retry only reruns vector search with the rewritten question --
        # rewording doesn't change which page-index section or SQL table an
        # LLM would pick the same way it changes embedding-similarity
        # results, so re-paying for the entire fan-out again on every retry
        # bought nothing but cost. Reuse what the first pass already found.
        docs = docs + state.get("_vectorless_results", [])

    return {"documents": docs}


def grade_documents(state: GraphState) -> dict[str, Any]:
    """One batched LLM call grading all retrieved docs at once (cheaper and
    faster than a call per document) -- returns the indices judged relevant
    to the question."""
    documents = state["documents"]
    if not documents:
        return {"documents": [], "no_relevant_docs": True}

    numbered = "\n\n".join(f"[{i}] {d['content'][:500]}" for i, d in enumerate(documents))
    prompt = (
        "You are grading whether retrieved passages are relevant to a user question.\n"
        f"Question: {state['question']}\n\n"
        f"Passages:\n{numbered}\n\n"
        "Reply with ONLY a comma-separated list of the relevant passage numbers "
        "(e.g. '0,2,3'). If none are relevant, reply with 'none'."
    )
    try:
        response = _strip_think(_Clients.utility_llm().invoke(prompt).content)
    except Exception as e:
        # Fail open: an API outage/rate-limit here shouldn't crash the whole
        # query (verified live: an exhausted Groq quota previously propagated
        # all the way up as an unhandled exception, showing the end user a
        # raw traceback). Pass every retrieved doc through ungraded rather
        # than discarding them -- generate()'s own error handling is the
        # backstop if the outage is total.
        print(f"[ERROR] Document grading failed, passing all retrieved docs through ungraded: {e}")
        return {"documents": documents, "no_relevant_docs": False}

    relevant_indices: set[int] = set()
    if response.strip().lower() != "none":
        for tok in response.replace(" ", "").split(","):
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(documents):
                    relevant_indices.add(idx)

    relevant_docs = [documents[i] for i in sorted(relevant_indices)]
    return {"documents": relevant_docs, "no_relevant_docs": len(relevant_docs) < MIN_RELEVANT_DOCS}


def transform_query(state: GraphState) -> dict[str, Any]:
    prompt = (
        "Rewrite the following question to be more effective for semantic "
        "search retrieval over a document corpus. Keep the same intent. "
        "Reply with ONLY the rewritten question, no explanation.\n\n"
        f"Original question: {state['question']}"
    )
    try:
        rewritten = _strip_think(_Clients.utility_llm().invoke(prompt).content).strip()
    except Exception as e:
        print(f"[ERROR] Query rewrite failed, retrying with the unmodified question: {e}")
        rewritten = ""
    return {"question": rewritten or state["question"], "retry_count": state["retry_count"] + 1}


def no_related_answer(state: GraphState) -> dict[str, Any]:
    """Terminal node reached when grade_documents found nothing relevant
    after exhausting retries. Per an explicit decision, this replaces the
    old web_search fallback entirely -- an answer is only ever returned
    from what's actually indexed (vector/vectorless RAG), never the open
    web, so "nothing relevant locally" ends the query here rather than
    reaching further out."""
    return {"generation": guardrails.NO_RELATED_ANSWER_MESSAGE, "sources": []}


def generate(state: GraphState) -> dict[str, Any]:
    documents = state["documents"]
    if not documents:
        return {
            "generation": "I don't have enough information in the available documents to answer this question.",
            "sources": [],
        }

    context = "\n\n".join(f"[{i}] {d['content']}" for i, d in enumerate(documents))
    prompt = (
        "Answer the question using ONLY the context below -- never use outside knowledge or "
        "anything not explicitly stated in this context, even if you know the answer from "
        "elsewhere. If the context doesn't contain the answer, say so explicitly rather than "
        "filling the gap yourself.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_question']}\n\nAnswer:"
    )
    try:
        answer = _strip_think(_Clients.generation_llm().invoke(prompt).content)
    except Exception as e:
        # Verified live: an exhausted Groq quota propagated as an unhandled
        # exception all the way to the Streamlit UI, rendering a raw Python
        # traceback to the end user instead of a message. Any backend LLM
        # outage must degrade to a plain-language response, never a crash.
        print(f"[ERROR] Answer generation failed: {e}")
        return {
            "generation": "I'm having trouble generating an answer right now (the language model service is unavailable). Please try again shortly.",
            "sources": [],
            "_skip_cache": True,
        }
    # Output-side PII redaction (Phase 3c) -- runs before the groundedness/
    # toxicity checks below so those judge the same text the user will see.
    answer = guardrails.redact_pii(answer)

    seen = set()
    sources = []
    for d in documents:
        meta = d["metadata"]
        key = (meta.get("source_file", ""), meta.get("page", -1))
        if key in seen:
            continue
        seen.add(key)
        try:
            images = json.loads(meta.get("images_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            images = []
        sources.append(
            {
                "source": meta.get("source_file", ""),
                "page": meta.get("page", -1),
                "file_type": meta.get("file_type", ""),
                "score": d.get("score"),
                # Diagrams attached to THIS chunk specifically -- a diagram
                # here may come from a different source document than the
                # text in a neighboring source entry (retrieval treats each
                # chunk independently), so always render it under its own
                # source citation, never implicitly attributed elsewhere.
                "images": images,
                # The actual (PII-redacted) retrieved text this citation is
                # based on -- not just a filename/page pointer. When an
                # answer draws on multiple sources, each one's full
                # contribution is inspectable on its own, not flattened
                # into one undifferentiated citation list.
                "content": guardrails.redact_pii(d["content"]),
            }
        )

    # Kept for output_guardrail's groundedness check, which needs the same
    # context the generation prompt saw -- not persisted in GraphState
    # since it's regenerable and only this one downstream node needs it.
    return {"generation": answer, "sources": sources, "_last_context": context}


def output_guardrail(state: GraphState) -> dict[str, Any]:
    """Phase 3c, after generate(): groundedness (is the answer actually
    supported by the retrieved context, not the model's own knowledge) and
    toxicity checks. Only reached from generate() -- a cache hit was
    already vetted when it was first generated, and no_related_answer /
    input_safety_guardrail's refusal are fixed safe strings, not
    LLM-authored free text pulled from context, so neither needs
    re-checking here."""
    answer = state["generation"]
    context = state.get("_last_context", "")

    is_grounded, ungrounded_message = guardrails.check_groundedness(answer, context)
    if not is_grounded:
        return {"generation": ungrounded_message, "sources": []}

    is_non_toxic, toxic_message = guardrails.check_output_toxicity(answer)
    if not is_non_toxic:
        return {"generation": toxic_message, "sources": []}

    return {}


# --- Routing -----------------------------------------------------------------

# Leaves headroom under the ~30s end-to-end target for generate() +
# output_guardrail to still run after this check fires.
QUERY_TIME_BUDGET_SECONDS = 25


def _route_after_input_guardrail(state: GraphState) -> str:
    return "blocked" if state["blocked"] else "check_cache"


def _route_after_cache_check(state: GraphState) -> str:
    return "end" if state["cache_hit"] else "retrieve"


def _route_after_grading(state: GraphState) -> str:
    if not state["no_relevant_docs"]:
        return "generate"
    elapsed = time.monotonic() - state.get("_query_start_time", 0.0)
    if state["retry_count"] < config.MAX_RETRIES and elapsed < QUERY_TIME_BUDGET_SECONDS:
        return "transform_query"
    return "no_related_answer"


def _route_after_output_guardrail(state: GraphState) -> str:
    return "skip" if state.get("_skip_cache") else "store"


# --- Graph assembly -----------------------------------------------------------------


def build_graph(checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("input_safety_guardrail", input_safety_guardrail)
    graph.add_node("check_cache", check_cache)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("no_related_answer", no_related_answer)
    graph.add_node("generate", generate)
    graph.add_node("output_guardrail", output_guardrail)
    graph.add_node("store_cache", store_cache)

    graph.add_edge(START, "input_safety_guardrail")
    graph.add_conditional_edges(
        "input_safety_guardrail",
        _route_after_input_guardrail,
        {"blocked": END, "check_cache": "check_cache"},
    )
    graph.add_conditional_edges(
        "check_cache", _route_after_cache_check, {"end": END, "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {"generate": "generate", "transform_query": "transform_query", "no_related_answer": "no_related_answer"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("no_related_answer", "store_cache")
    graph.add_edge("generate", "output_guardrail")
    graph.add_conditional_edges(
        "output_guardrail",
        _route_after_output_guardrail,
        {"store": "store_cache", "skip": END},
    )
    graph.add_edge("store_cache", END)

    return graph.compile(checkpointer=checkpointer)


def build_initial_state(question: str, target_document: Optional[str] = None) -> GraphState:
    """Single source of truth for a fresh GraphState -- used by run_query
    below and by the Streamlit Chat page, so GraphState's fields (already
    changed shape three times across Phases 3/3b/3c) only ever need
    updating in one place."""
    return {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "no_relevant_docs": False,
        "retry_count": 0,
        "sources": [],
        "cache_hit": False,
        "target_document": target_document,
        "blocked": False,
        "_last_context": "",
        "_skip_cache": False,
        "_vectorless_results": [],
        "_query_start_time": 0.0,
    }


_compiled_graph = None


def run_query(question: str, target_document: Optional[str] = None) -> dict[str, Any]:
    """Convenience wrapper: builds (once, cached) and invokes the graph.

    target_document: optional filename to scope retrieval to one known
    document (Phase 3b's SQL table / page-index tree / vector source_file
    filter) instead of searching everything. No UI exists yet to set this
    from Phase 4, so it defaults to None ("search everything")."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph.invoke(build_initial_state(question, target_document))


if __name__ == "__main__":
    result = run_query("What is Deepak's experience?")
    print("\n=== Generation ===")
    print(result["generation"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(s)
