"""LangGraph orchestration: a Corrective/Adaptive RAG flow.

retrieve -> grade_documents -> [generate | transform_query (retry loop) | web_search (fallback)] -> generate

See Phase 3 in the implementation plan for the design rationale.
"""

import re
from typing import Any, Optional, TypedDict

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from rag_learn import config
from rag_learn.embedding import EmbeddingPipeline
from rag_learn.search import Retriever
from rag_learn.vectorstore import VectorStore

# Grading/query-rewrite are short structured tasks -- reasoning_effort="none"
# skips the model's <think> block entirely, which otherwise burns output
# tokens on internal deliberation nobody reads for a yes/no or one-line
# rewrite. Kept separate from the generation LLM in case that one is ever
# tuned differently (e.g. more reasoning for complex synthesis).
_UTILITY_LLM_KWARGS = {"temperature": 0.0, "reasoning_effort": "none"}
_GENERATION_LLM_KWARGS = {"temperature": 0.1, "reasoning_effort": "none"}

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
    web_search_needed: bool
    retry_count: int
    sources: list[dict[str, Any]]


class _Clients:
    """Lazily-initialized singletons shared across graph invocations in this
    process, so the embedding model and vector store connection are loaded
    once, not per query."""

    _pipeline: Optional[EmbeddingPipeline] = None
    _store: Optional[VectorStore] = None
    _retriever: Optional[Retriever] = None
    _utility_llm: Optional[ChatGroq] = None
    _generation_llm: Optional[ChatGroq] = None
    _tavily: Optional[TavilySearch] = None

    @classmethod
    def retriever(cls) -> Retriever:
        if cls._retriever is None:
            cls._pipeline = cls._pipeline or EmbeddingPipeline()
            cls._store = cls._store or VectorStore()
            cls._retriever = Retriever(cls._store, cls._pipeline)
        return cls._retriever

    @classmethod
    def utility_llm(cls) -> ChatGroq:
        if cls._utility_llm is None:
            cls._utility_llm = ChatGroq(model_name=config.GROQ_MODEL_NAME, **_UTILITY_LLM_KWARGS)
        return cls._utility_llm

    @classmethod
    def generation_llm(cls) -> ChatGroq:
        if cls._generation_llm is None:
            cls._generation_llm = ChatGroq(model_name=config.GROQ_MODEL_NAME, **_GENERATION_LLM_KWARGS)
        return cls._generation_llm

    @classmethod
    def tavily(cls) -> TavilySearch:
        if cls._tavily is None:
            cls._tavily = TavilySearch(max_results=5)
        return cls._tavily


# --- Nodes -----------------------------------------------------------------


def retrieve(state: GraphState) -> dict[str, Any]:
    docs = _Clients.retriever().retrieve(
        state["question"], top_k=config.TOP_K, score_threshold=config.SCORE_THRESHOLD
    )
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict[str, Any]:
    """One batched LLM call grading all retrieved docs at once (cheaper and
    faster than a call per document) -- returns the indices judged relevant
    to the question."""
    documents = state["documents"]
    if not documents:
        return {"documents": [], "web_search_needed": True}

    numbered = "\n\n".join(f"[{i}] {d['content'][:500]}" for i, d in enumerate(documents))
    prompt = (
        "You are grading whether retrieved passages are relevant to a user question.\n"
        f"Question: {state['question']}\n\n"
        f"Passages:\n{numbered}\n\n"
        "Reply with ONLY a comma-separated list of the relevant passage numbers "
        "(e.g. '0,2,3'). If none are relevant, reply with 'none'."
    )
    response = _strip_think(_Clients.utility_llm().invoke(prompt).content)

    relevant_indices: set[int] = set()
    if response.strip().lower() != "none":
        for tok in response.replace(" ", "").split(","):
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(documents):
                    relevant_indices.add(idx)

    relevant_docs = [documents[i] for i in sorted(relevant_indices)]
    return {"documents": relevant_docs, "web_search_needed": len(relevant_docs) < MIN_RELEVANT_DOCS}


def transform_query(state: GraphState) -> dict[str, Any]:
    prompt = (
        "Rewrite the following question to be more effective for semantic "
        "search retrieval over a document corpus. Keep the same intent. "
        "Reply with ONLY the rewritten question, no explanation.\n\n"
        f"Original question: {state['question']}"
    )
    rewritten = _strip_think(_Clients.utility_llm().invoke(prompt).content).strip()
    return {"question": rewritten or state["question"], "retry_count": state["retry_count"] + 1}


def web_search(state: GraphState) -> dict[str, Any]:
    try:
        result = _Clients.tavily().invoke({"query": state["question"]})
        web_docs = [
            {
                "content": r.get("content", ""),
                "metadata": {"source_file": r.get("url", ""), "file_type": "web", "page": -1},
                "score": None,
            }
            for r in result.get("results", [])
            if r.get("content")
        ]
    except Exception as e:
        print(f"[ERROR] Web search failed: {e}")
        web_docs = []
    return {"documents": state["documents"] + web_docs}


def generate(state: GraphState) -> dict[str, Any]:
    documents = state["documents"]
    if not documents:
        return {
            "generation": "I don't have enough information in the available documents or web search results to answer this question.",
            "sources": [],
        }

    context = "\n\n".join(f"[{i}] {d['content']}" for i, d in enumerate(documents))
    prompt = (
        "Use the following context to answer the question concisely. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_question']}\n\nAnswer:"
    )
    answer = _strip_think(_Clients.generation_llm().invoke(prompt).content)

    seen = set()
    sources = []
    for d in documents:
        meta = d["metadata"]
        key = (meta.get("source_file", ""), meta.get("page", -1))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": meta.get("source_file", ""),
                "page": meta.get("page", -1),
                "file_type": meta.get("file_type", ""),
                "score": d.get("score"),
            }
        )

    return {"generation": answer, "sources": sources}


# --- Routing -----------------------------------------------------------------


def _route_after_grading(state: GraphState) -> str:
    if not state["web_search_needed"]:
        return "generate"
    if state["retry_count"] < config.MAX_RETRIES:
        return "transform_query"
    return "web_search"


# --- Graph assembly -----------------------------------------------------------------


def build_graph(checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


_compiled_graph = None


def run_query(question: str) -> dict[str, Any]:
    """Convenience wrapper: builds (once, cached) and invokes the graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "web_search_needed": False,
        "retry_count": 0,
        "sources": [],
    }
    return _compiled_graph.invoke(initial_state)


if __name__ == "__main__":
    result = run_query("What is Deepak's experience?")
    print("\n=== Generation ===")
    print(result["generation"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(s)
