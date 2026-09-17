# Search-Everything Retrieval Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut unscoped ("Search everything") Chat query latency from 4.5+ minutes to under 30 seconds, remove reliance on the paid Anthropic fallback, and let answers synthesize across multiple documents instead of collapsing to one "best" source.

**Architecture:** Replace the per-tree LLM relevance check (one call per page-index document, ~45+ calls per query) with a free local-embedding pre-filter followed by one batched LLM call across the shortlist. Skip the slow free-model fallback specifically at that high-volume call site. Stop re-running the whole vectorless fan-out on every Corrective-RAG retry, and add a hard time ceiling as a backstop.

**Tech Stack:** Python 3.14, LangChain/LangGraph, Groq (`qwen/qwen3.6-27b`), `sentence-transformers` (already a dependency), DuckDB (telemetry, for verification only). No test framework is configured in this repo (per `CLAUDE.md`) — verification here follows the codebase's existing convention of small live functional-check scripts (see the many "verified live" comments throughout `src/rag_learn/`), not pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-search-everything-retrieval-speed-design.md`

## Global Constraints

- Unscoped queries must answer in ≤30 seconds (spec Goals).
- No paid LLM tier — `ANTHROPIC_API_KEY` is being removed from `.env`; `config.get_llm()`/`get_independent_judge_llm()` already degrade gracefully with no key set (no code change needed for that fact itself).
- "Search this document" (scoped) queries must be unaffected — same behavior, same speed as today.
- The SQL vectorless path and ingestion/chunking/embedding pipeline are out of scope — do not touch `vectorless_sql.py`, `data_loader.py`, `embedding.py`, `sync.py`.
- Every new/changed LLM call site keeps this codebase's fail-open convention: a call failure never crashes the query, it degrades to a safe default and prints an `[ERROR]`-prefixed message, matching existing style in `guardrails.py`/`graph.py`.
- No new external dependencies — `sentence-transformers`, `langchain-groq`, `numpy` are already in `pyproject.toml`.

---

### Task 1: Groq-only LLM constructor (no fallback chain)

**Files:**
- Modify: `src/rag_learn/config.py` (add after `get_llm`, i.e. after line 242, before `get_independent_judge_llm`)
- Test: none needed as a separate file — verify with a one-off script (Step 2 below), matching this repo's convention (no `tests/` directory exists).

**Interfaces:**
- Produces: `get_llm_groq_only(temperature: float = 0.0, max_tokens: "int | None" = None, purpose: str = "unspecified")` — returns a Groq-only `Runnable` (no `.with_fallbacks()`), tagged `"provider:groq"`, with a `TelemetryCallback` attached. Used by Task 3.

- [ ] **Step 1: Add the function to `config.py`**

Insert immediately after `get_llm`'s closing line (currently line 242, the `return llm.with_config(...)` line of `get_llm`) and before `def get_independent_judge_llm(...)`:

```python
def get_llm_groq_only(temperature: float = 0.0, max_tokens: "int | None" = None, purpose: str = "unspecified"):
    """Groq-only construction with NO fallback chain -- for a call site that
    fires many times per query in a short burst (see vectorless_pageindex's
    batched relevance pick). Verified live: a bursty volume of calls
    saturates the free OpenRouter fallback tier just as badly as Groq
    itself, and waiting 10+ seconds per call on a free model that ignores
    "answer with one number" instructions costs more than it's worth for a
    cheap relevance check. Callers at this call site must handle a failure
    themselves (fail open toward inclusion, not exclusion) rather than
    trusting a slow fallback to save them."""
    from langchain_groq import ChatGroq

    from rag_learn import telemetry

    llm = ChatGroq(**llm_kwargs(temperature=temperature, max_tokens=max_tokens)).with_config(
        {"tags": ["provider:groq"]}
    )
    return llm.with_config({"callbacks": [telemetry.TelemetryCallback(purpose=purpose)]})
```

- [ ] **Step 2: Verify it works and has no fallback chain**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
from rag_learn import config
llm = config.get_llm_groq_only(max_tokens=10, purpose='plan_verify_groq_only')
print(type(llm).__name__)
result = llm.invoke('Reply with exactly the word OK')
print(repr(result.content))
"
```
Expected: prints a type name that is NOT `RunnableWithFallbacks` (should be a `RunnableBinding`/`ChatGroq`-derived type), and prints a response containing `OK`.

- [ ] **Step 3: Commit**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn
git add src/rag_learn/config.py
git commit -m "Add Groq-only LLM constructor for high-volume call sites"
```

---

### Task 2: Local embedding pre-filter for page-index trees

**Files:**
- Modify: `src/rag_learn/vectorless_pageindex.py` (add near the top, after the existing lazy-singleton LLM getters, i.e. after line 62's `_get_pageindex_picker_llm`)

**Interfaces:**
- Consumes: `config.EMBEDDING_MODEL` (existing), `_load_tree(tree_path)` (existing, line 337-338).
- Produces: `_prefilter_trees(question: str, tree_paths: List[Path], top_n: int = _TOP_N_CANDIDATES) -> List[Path]`. Used by Task 4.

- [ ] **Step 1: Add the embedder singleton and prefilter function**

Insert after `_get_pageindex_picker_llm` (after line 62):

```python
_embedder = None

_TOP_N_CANDIDATES = 8


def _get_embedder():
    """Lazy singleton, same pattern as the LLM getters above -- avoids
    loading the (~1.5GB) embedding model for a query that never needs the
    prefilter (e.g. a scoped single-document query)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _prefilter_trees(question: str, tree_paths: List[Path], top_n: int = _TOP_N_CANDIDATES) -> List[Path]:
    """Local, free, no-LLM-call narrowing of which trees are even worth an
    LLM look -- cosine similarity between the question and each tree's own
    top-level section summaries (already written at ingestion), using the
    same embedding model already loaded for vector search. Verified live:
    firing an LLM call per tree (45+ trees in this corpus) was the actual
    bottleneck for an unscoped query (4.5+ minutes, ~130 calls); this step
    is a single local encode pass plus a numpy comparison, effectively
    free and near-instant."""
    if len(tree_paths) <= top_n:
        return tree_paths
    import numpy as np

    model = _get_embedder()
    question_embedding = model.encode([question])[0]

    scored = []
    for tp in tree_paths:
        tree = _load_tree(tp)
        summary_text = " ".join(s.get("summary") or "" for s in tree.get("sections", []))
        if not summary_text.strip():
            scored.append((tp, 0.0))
            continue
        tree_embedding = model.encode([summary_text])[0]
        similarity = float(
            np.dot(question_embedding, tree_embedding)
            / (np.linalg.norm(question_embedding) * np.linalg.norm(tree_embedding) + 1e-9)
        )
        scored.append((tp, similarity))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [tp for tp, _ in scored[:top_n]]
```

- [ ] **Step 2: Verify it ranks semantically-related trees higher**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
import json, tempfile
from pathlib import Path
from rag_learn.vectorless_pageindex import _prefilter_trees

tmp = Path(tempfile.mkdtemp())
trees = {
    'llm.json': {'source_file': 'llm.json', 'sections': [{'title': 'Tokens', 'summary': 'Explains how large language models break text into tokens for processing.'}]},
    'cooking.json': {'source_file': 'cooking.json', 'sections': [{'title': 'Recipes', 'summary': 'A collection of pasta and bread recipes for home cooking.'}]},
    'gardening.json': {'source_file': 'gardening.json', 'sections': [{'title': 'Soil', 'summary': 'Tips for improving garden soil quality and composting.'}]},
}
paths = []
for name, content in trees.items():
    p = tmp / name
    p.write_text(json.dumps(content))
    paths.append(p)

result = _prefilter_trees('What are tokens in an LLM?', paths, top_n=1)
print([p.name for p in result])
assert result[0].name == 'llm.json', f'expected llm.json first, got {result[0].name}'
print('PASS')
"
```
Expected: prints `['llm.json']` then `PASS`.

- [ ] **Step 3: Commit**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn
git add src/rag_learn/vectorless_pageindex.py
git commit -m "Add local-embedding prefilter for page-index tree fan-out"
```

---

### Task 3: Batched multi-section relevance pick

**Files:**
- Modify: `src/rag_learn/vectorless_pageindex.py` (add after `_pick_section`, i.e. after line 372, before `_query_tree`)

**Interfaces:**
- Consumes: `config.get_llm_groq_only` (Task 1).
- Produces: `_pick_sections_batch(question: str, sections: List[Dict[str, Any]]) -> List[int]`. Used by Task 4.

- [ ] **Step 1: Add the batch LLM singleton and the function**

Insert after `_pick_section`'s closing (after line 372) and before `def _query_tree(...)`:

```python
_pageindex_batch_llm = None


def _get_pageindex_batch_llm():
    global _pageindex_batch_llm
    if _pageindex_batch_llm is None:
        # No fallback chain (get_llm_groq_only) -- see _pick_sections_batch's
        # docstring for why a slow free-model fallback isn't worth it here.
        _pageindex_batch_llm = config.get_llm_groq_only(
            temperature=0.0, max_tokens=60, purpose="pageindex_batch_pick"
        )
    return _pageindex_batch_llm


def _pick_sections_batch(question: str, sections: List[Dict[str, Any]]) -> List[int]:
    """Batched version of _pick_section: one LLM call judging ALL candidate
    sections at once (mirrors graph.grade_documents' batching), returning
    EVERY plausibly relevant index rather than a single best match -- a
    broad/definitional question (e.g. "What is RAG?") legitimately has a
    paragraph-level answer spread across several different documents rather
    than being any one document's dedicated chapter, and keeping only one
    match systematically under-serves that case. On failure, keeps every
    candidate rather than waiting on a slow fallback (see
    get_llm_groq_only's docstring)."""
    if not sections:
        return []
    listing = "\n".join(f"[{i}] {s['title']}: {s['summary']}" for i, s in enumerate(sections))
    prompt = (
        "Given the question and this list of document sections, respond with ONLY a "
        "comma-separated list of the bracketed index numbers of every section that looks "
        "at least plausibly relevant (e.g. `0,2,3`), or `none` if none are. Err toward "
        "including a section if it's a reasonable candidate -- a downstream step will "
        "double check. Each section's index is the number in [brackets] at the start of "
        "its line -- ignore any other numbers inside a section's own title or summary.\n\n"
        f"Sections:\n{listing}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        answer = _get_pageindex_batch_llm().invoke(prompt).content.strip().lower()
    except Exception as e:
        print(f"[ERROR] Page-index batch section selection failed, keeping all candidates: {e}")
        return list(range(len(sections)))
    if answer == "none":
        return []
    indices: set[int] = set()
    for tok in answer.replace(" ", "").split(","):
        digits = re.sub(r"[^\d]", "", tok)
        if digits and 0 <= int(digits) < len(sections):
            indices.add(int(digits))
    return sorted(indices)
```

- [ ] **Step 2: Verify it selects relevant sections and excludes irrelevant ones**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
from rag_learn.vectorless_pageindex import _pick_sections_batch

sections = [
    {'title': 'Tokenization', 'summary': 'How LLMs split text into tokens before processing.'},
    {'title': 'Pasta recipes', 'summary': 'Classic Italian pasta dishes and sauces.'},
    {'title': 'Token limits', 'summary': 'Why models have a maximum context window measured in tokens.'},
    {'title': 'Garden composting', 'summary': 'How to build a compost bin for garden soil.'},
]
result = _pick_sections_batch('What are tokens?', sections)
print(result)
assert 0 in result and 2 in result, f'expected indices 0 and 2 (the LLM-token sections), got {result}'
print('PASS')
"
```
Expected: prints a list containing `0` and `2`, then `PASS`. (The LLM call is real and non-deterministic in wording, but should reliably identify the two token-related sections.)

- [ ] **Step 3: Commit**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn
git add src/rag_learn/vectorless_pageindex.py
git commit -m "Add batched multi-section relevance pick for page-index queries"
```

---

### Task 4: Wire prefilter + batched pick into `query()`

**Files:**
- Modify: `src/rag_learn/vectorless_pageindex.py:414-427` (the `query` function)

**Interfaces:**
- Consumes: `_prefilter_trees` (Task 2), `_pick_sections_batch` (Task 3), existing `_pick_section`/`_query_tree`/`_load_tree`/`list_trees`.
- Produces: `query(question, tree_paths=None)` — same signature as before, but the unscoped path (`tree_paths=None`, i.e. "search everything") now returns results for potentially multiple documents/sections instead of at most one per tree, via the fast batched path. The scoped path (`tree_paths` given, used by `graph._match_pageindex_tree`) is unchanged (still uses `_query_tree`).

- [ ] **Step 1: Replace the `query` function**

Replace the existing `query` function (lines 414-427) with:

```python
def query(question: str, tree_paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Navigate page-index trees and return matched sections' full text,
    shaped like the vector path's output so graph.py's generate() node can
    consume either uniformly.

    Scoped call (tree_paths given, from graph._match_pageindex_tree's
    "search this document" mode): unchanged behavior, one tree, the
    existing single-pick walk (_query_tree).

    Unscoped call ("search everything", tree_paths=None): this is the path
    that was observed live to fire 45+ individual LLM calls and take 4.5+
    minutes. Replaced with: a free local-embedding prefilter to a shortlist
    (_prefilter_trees), then ONE batched LLM call across that shortlist's
    top-level sections (_pick_sections_batch) returning every plausibly
    relevant match, not just one -- letting several different documents'
    perspectives on the same topic survive into generate()."""
    candidates = tree_paths if tree_paths is not None else list_trees()
    if not candidates:
        return []

    if tree_paths is not None:
        with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
            results = pool.map(lambda tp: _query_tree(question, tp), candidates)
        return [r for r in results if r is not None]

    shortlisted = _prefilter_trees(question, candidates)
    trees = {tp: _load_tree(tp) for tp in shortlisted}
    top_sections_by_tree = {tp: t.get("sections", []) for tp, t in trees.items()}

    # One flat listing across every shortlisted tree's top-level sections,
    # not one call per tree -- see _pick_sections_batch's docstring. Track
    # which (tree, local_index) each flattened entry came from so the
    # response's indices can be routed back to their source tree.
    flat_sections: List[Dict[str, Any]] = []
    origin: List[tuple[Path, int]] = []
    for tp, sections in top_sections_by_tree.items():
        for i, s in enumerate(sections):
            flat_sections.append(s)
            origin.append((tp, i))

    matched_indices = _pick_sections_batch(question, flat_sections)

    results: List[Dict[str, Any]] = []
    for idx in matched_indices:
        tp, local_i = origin[idx]
        section = top_sections_by_tree[tp][local_i]
        tree = trees[tp]
        children = section.get("children")
        if children:
            # Bounded, low-volume (at most len(shortlisted) of these) --
            # the existing single-pick walk is fine at this depth.
            child_idx = _pick_section(question, children)
            section = children[child_idx] if child_idx is not None else section
        results.append(
            {
                "content": section["text"],
                "metadata": {
                    "source_file": tree.get("source_file", tp.name),
                    "file_type": "pageindex_section",
                    "page": section.get("page_start", -1),
                    "section_title": section["title"],
                },
                "score": None,
            }
        )
    return results
```

- [ ] **Step 2: Verify against the real corpus — speed and multi-source result**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
import time
from rag_learn.vectorless_pageindex import query

t0 = time.monotonic()
results = query('What are tokens?')
elapsed = time.monotonic() - t0
sources = sorted({r['metadata']['source_file'] for r in results})
print(f'elapsed={elapsed:.1f}s, results={len(results)}, sources={sources}')
assert elapsed < 20, f'expected under 20s, took {elapsed:.1f}s'
print('PASS')
"
```
Expected: `elapsed` well under 20 seconds (vs. 4.5+ minutes before), `results` non-empty, `sources` ideally showing more than one distinct document, then `PASS`.

- [ ] **Step 3: Commit**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn
git add src/rag_learn/vectorless_pageindex.py
git commit -m "Wire prefilter + batched pick into unscoped page-index query"
```

---

### Task 5: Bound the Corrective-RAG retry loop's cost

**Files:**
- Modify: `src/rag_learn/graph.py:1-30` (imports), `:42-75` (GraphState), `:135-156` (input_safety_guardrail), `:194-237` (retrieve), `:403-420` (routing), `:466-484` (build_initial_state)

**Interfaces:**
- Consumes: nothing new externally.
- Produces: `GraphState` gains `_vectorless_results: list[dict[str, Any]]` and `_query_start_time: float`. `_route_after_grading` now also stops retrying once a time budget is exceeded, independent of `retry_count`.

- [ ] **Step 1: Add the `time` import**

At the top of `graph.py`, alongside the existing `import json` / `import re` (lines 15-16), add:

```python
import time
```

- [ ] **Step 2: Add the two new GraphState fields**

In the `GraphState` TypedDict (lines 42-75), add after the existing `_skip_cache: bool` field (the last field, line 74):

```python
    # First-pass vectorless (SQL + page-index) results, cached so a
    # transform_query retry doesn't re-pay for the entire fan-out -- see
    # retrieve()'s retry branch below.
    _vectorless_results: list[dict[str, Any]]
    # time.monotonic() at input_safety_guardrail -- lets _route_after_grading
    # enforce a hard time budget regardless of retry_count (see
    # QUERY_TIME_BUDGET_SECONDS below).
    _query_start_time: float
```

- [ ] **Step 3: Set `_query_start_time` in `input_safety_guardrail`**

In `input_safety_guardrail` (lines 135-156), both return statements need `"_query_start_time": time.monotonic()` added. The function becomes:

```python
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
```

- [ ] **Step 4: Skip re-running the vectorless fan-out on retry, in `retrieve`**

Replace the `retrieve` function (lines 194-237) with:

```python
def retrieve(state: GraphState) -> dict[str, Any]:
    target = state.get("target_document")

    if target:
        table = _match_sql_table(target)
        if table:
            return {"documents": vectorless_sql.query(state["question"], tables=[table])}
        tree_path = _match_pageindex_tree(target)
        if tree_path:
            return {"documents": vectorless_pageindex.query(state["question"], tree_paths=[tree_path])}
        # target_document isn't a known SQL table or page-index tree --
        # assume it's a vector-routed source_file and scope vector search
        # to just that document's chunks (falls through below).

    candidates = _Clients.retriever().retrieve(
        state["question"],
        top_k=config.RETRIEVE_CANDIDATES,
        score_threshold=config.SCORE_THRESHOLD,
        source_file=target,
    )
    docs = _Clients.reranker().rerank(state["question"], candidates, top_k=config.TOP_K)

    if not target:
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
```

- [ ] **Step 5: Add the hard time budget to `_route_after_grading`**

Replace the routing section (lines 403-420) — specifically add a constant before it and replace `_route_after_grading`:

```python
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
```

- [ ] **Step 6: Add the two new fields to `build_initial_state`**

In `build_initial_state` (lines 466-484), add after `"_skip_cache": False,`:

```python
        "_vectorless_results": [],
        "_query_start_time": 0.0,
```

- [ ] **Step 7: Verify the time-budget routing logic in isolation**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
import time
from rag_learn.graph import _route_after_grading, build_initial_state

state = build_initial_state('test question')
state['no_relevant_docs'] = True
state['retry_count'] = 0

# Fresh start time -- should still retry.
state['_query_start_time'] = time.monotonic()
result = _route_after_grading(state)
print('fresh:', result)
assert result == 'transform_query'

# Stale start time (past the budget) -- should stop retrying even though
# retry_count is still under the max.
state['_query_start_time'] = time.monotonic() - 999
result = _route_after_grading(state)
print('stale:', result)
assert result == 'no_related_answer'
print('PASS')
"
```
Expected: `fresh: transform_query`, `stale: no_related_answer`, `PASS`.

- [ ] **Step 8: Verify the app still starts (import/syntax sanity)**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "from rag_learn.graph import build_graph; build_graph(); print('graph builds OK')"
```
Expected: `graph builds OK` with no traceback.

- [ ] **Step 9: Commit**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn
git add src/rag_learn/graph.py
git commit -m "Bound retry loop: reuse first-pass vectorless results, add time budget"
```

---

### Task 6: Skip redundant re-grading of already-vetted page-index results

**Files:**
- Modify: `src/rag_learn/graph.py:240-277` (`grade_documents`)

**Interfaces:**
- Consumes: `d["metadata"]["file_type"] == "pageindex_section"` tag (already set by `vectorless_pageindex.query`, unchanged).
- Produces: `grade_documents` keeps the same return shape (`{"documents": ..., "no_relevant_docs": ...}`).

- [ ] **Step 1: Replace `grade_documents`**

Replace the existing function (lines 240-277) with:

```python
def grade_documents(state: GraphState) -> dict[str, Any]:
    """One batched LLM call grading retrieved docs at once (cheaper and
    faster than a call per document) -- returns the indices judged relevant
    to the question.

    Page-index-sourced documents (file_type == "pageindex_section") skip
    this grading entirely: they already passed a relevance judgment inside
    vectorless_pageindex.query() (_pick_sections_batch) at ingestion-query
    time, so re-grading them here is redundant cost, and a stricter second
    pass was observed to discard legitimate partial matches for broad
    questions where the right answer draws on several documents at once
    rather than one dominant source -- see the design spec's "loosen
    relevance retention" goal."""
    documents = state["documents"]
    if not documents:
        return {"documents": [], "no_relevant_docs": True}

    pre_approved = [d for d in documents if d.get("metadata", {}).get("file_type") == "pageindex_section"]
    to_grade = [d for d in documents if d.get("metadata", {}).get("file_type") != "pageindex_section"]

    if not to_grade:
        return {"documents": pre_approved, "no_relevant_docs": len(pre_approved) < MIN_RELEVANT_DOCS}

    numbered = "\n\n".join(f"[{i}] {d['content'][:500]}" for i, d in enumerate(to_grade))
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
        print(f"[ERROR] Document grading failed, passing all retrieved docs through ungraded: {e}")
        return {"documents": pre_approved + to_grade, "no_relevant_docs": False}

    relevant_indices: set[int] = set()
    if response.strip().lower() != "none":
        for tok in response.replace(" ", "").split(","):
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(to_grade):
                    relevant_indices.add(idx)

    graded_relevant = [to_grade[i] for i in sorted(relevant_indices)]
    relevant_docs = pre_approved + graded_relevant
    return {"documents": relevant_docs, "no_relevant_docs": len(relevant_docs) < MIN_RELEVANT_DOCS}
```

- [ ] **Step 2: Verify pageindex-sourced docs always survive grading**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
from rag_learn.graph import grade_documents

state = {
    'question': 'What are tokens?',
    'documents': [
        {'content': 'Tokens are units of text an LLM processes.', 'metadata': {'file_type': 'pageindex_section', 'source_file': 'a.pdf'}},
        {'content': 'Completely unrelated pasta recipe content.', 'metadata': {'file_type': 'pdf', 'source_file': 'b.pdf'}},
    ],
}
result = grade_documents(state)
sources = [d['metadata']['source_file'] for d in result['documents']]
print(sources, result['no_relevant_docs'])
assert 'a.pdf' in sources, 'pageindex_section doc must always survive grading'
print('PASS')
"
```
Expected: `a.pdf` is present in `sources` regardless of whether `b.pdf` (the real Groq grading call for the non-pageindex doc) survives, then `PASS`.

- [ ] **Step 3: Commit**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn
git add src/rag_learn/graph.py
git commit -m "Skip redundant re-grading of already-vetted page-index results"
```

---

### Task 7: Remove the Anthropic fallback tier

**Files:**
- Modify: `.env`

**Interfaces:** none (env-only change; `config.py` already handles an absent key gracefully).

- [ ] **Step 1: Read the current `.env` and remove/blank the Anthropic key**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && grep -n ANTHROPIC .env
```

Then edit `.env` to comment out (or delete) the `ANTHROPIC_API_KEY=...` line, e.g. replace it with:

```
# ANTHROPIC_API_KEY=  # removed intentionally -- paid fallback no longer used (see docs/superpowers/specs/2026-09-14-search-everything-retrieval-speed-design.md)
```

- [ ] **Step 2: Verify the fallback chain drops to 2 tiers**

Run:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
from rag_learn import config
print('ANTHROPIC_API_KEY:', config.ANTHROPIC_API_KEY)
assert config.ANTHROPIC_API_KEY is None
judge = config.get_independent_judge_llm()
print('independent judge:', judge)
assert judge is None
print('PASS')
"
```
Expected: `ANTHROPIC_API_KEY: None`, `independent judge: None`, `PASS`.

- [ ] **Step 3: Commit**

Note: `.env` is gitignored (per `CLAUDE.md`), so this step will not stage anything — that's expected, skip committing this one.

---

### Task 8: End-to-end verification (the manual test pass)

**Files:** none modified — this task only runs and observes the app.

- [ ] **Step 1: Restart the Streamlit server so it picks up the new code**

The currently-running server (background task `bi9fgcebi`, started earlier this session) has the old module code cached in its process (`st.cache_resource`). Stop it and start a fresh one:

```bash
pkill -f "streamlit run streamlit_app.py"
```

Then start it again (background):
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run streamlit run streamlit_app.py --server.headless true
```

- [ ] **Step 2: Run the two known-slow questions via the CLI (faster to inspect than the UI) and confirm the time budget**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
import time
from rag_learn.graph import run_query

for q in ['What is RAG?', 'What are tokens?']:
    t0 = time.monotonic()
    result = run_query(q)
    elapsed = time.monotonic() - t0
    sources = sorted({s['source'] for s in result['sources']})
    print(f'{q!r}: {elapsed:.1f}s, {len(sources)} source(s): {sources}')
    print('  answer:', result['generation'][:200])
    assert elapsed < 30, f'{q!r} took {elapsed:.1f}s, over the 30s budget'
print('ALL PASS')
"
```
Expected: both questions complete in under 30 seconds each, ideally drawing on more than one source document, `ALL PASS` at the end.

- [ ] **Step 2b: Confirm no Anthropic calls were made and page-index call volume dropped**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
import duckdb
conn = duckdb.connect('data/vector_store/telemetry.duckdb', read_only=True)
rows = conn.execute(\"SELECT provider, purpose, COUNT(*) FROM llm_calls WHERE ts >= now() - INTERVAL 5 MINUTE GROUP BY provider, purpose ORDER BY provider, purpose\").fetchall()
for r in rows: print(r)
anthropic_calls = [r for r in rows if r[0] == 'anthropic']
assert not anthropic_calls, f'unexpected anthropic calls: {anthropic_calls}'
print('PASS: no anthropic calls')
"
```
Expected: no row with provider `anthropic`; `pageindex_batch_pick` purpose present with a low call count (single digits, not 45+); `PASS: no anthropic calls`.

- [ ] **Step 3: Confirm scoped ("search this document") queries are unaffected**

```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
from rag_learn.sync import list_indexed_documents
indexed = list_indexed_documents()
pageindex_docs = indexed.get('vectorless_pageindex', [])
print('a page-indexed doc to scope to:', pageindex_docs[0] if pageindex_docs else None)
"
```
Then, using the printed filename:
```bash
cd /Users/dludhani/Development_Work/Projects/RAG_Learn && unset VIRTUAL_ENV && uv run python3 -c "
import time
from rag_learn.graph import run_query
t0 = time.monotonic()
result = run_query('What is this document about?', target_document='<PASTE_FILENAME_HERE>')
print(f'{time.monotonic() - t0:.1f}s')
print(result['generation'][:200])
"
```
Expected: fast (well under 10s, unchanged from before this plan), single-document behavior.

- [ ] **Step 4: Hand off**

Confirm the Streamlit server from Step 1 is still running and reachable at `http://localhost:8501`, then report the verification results (timings, source counts, telemetry check) to the user so they can take over testing in the browser themselves.

---

## Self-Review Notes

- **Spec coverage:** all 6 design-doc items map to tasks — item 1 (prefilter) → Task 2, item 2 (batch) → Task 3-4, item 3 (drop fallback at this call site) → Task 1 + 3, item 4 (remove Anthropic) → Task 7, item 5 (bound retry cost) → Task 5, item 6 (loosen relevance retention) → Task 6.
- **No test framework exists in this repo** (confirmed via `CLAUDE.md`) — every verification step above is a real, runnable script consistent with this codebase's existing "verified live" convention, not a placeholder.
- **Type/interface consistency checked:** `_pick_sections_batch` return type (`List[int]`) is consumed identically in Task 4's `query()`; `get_llm_groq_only`'s signature matches `get_llm`'s existing signature shape so it's a drop-in swap at the one call site that uses it (Task 3).
