# Search-Everything Retrieval Speed & Quality — Design

## Problem

Unscoped ("Search everything") Chat queries fan out to vector search + every
SQL table + every `vectorless_pageindex` tree (currently 45 trees). Each tree
requires one or more individual LLM calls (`_pick_section`) to judge
relevance, capped at 4 concurrent workers. Live trace analysis (telemetry.duckdb,
queries "What is RAG?" and "What are tokens?") showed ~130 LLM calls and
4.5+ minutes for a single unscoped question, with only one actual `generation`
call in that whole window — nearly all the time is spent asking "is this one
document relevant?" one at a time.

Under that call volume, Groq's per-minute rate limit is exceeded, which
triggers fallback to the two free OpenRouter models; those get saturated too
under the same burst, cascading to paid Anthropic calls as a last resort —
real cost spent on a low-value relevance check, not the answer itself.

The Corrective RAG retry loop (`grade_documents` → `transform_query` →
`retrieve`, up to `RAG_MAX_RETRIES=2`) re-runs the entire fan-out from scratch
on a miss, multiplying this cost up to 3x for one question.

There's also a quality mismatch with the app's actual purpose: page-index
navigation picks a single best section per tree, and `grade_documents`
discards anything not judged clearly relevant. For broad/definitional
questions ("What is RAG?", "What are tokens?") that appear as a paragraph
across many documents rather than being any one document's dedicated chapter,
this systematically under-selects — the opposite of the goal, which is to
synthesize how several different documents define/explain the same topic
differently.

## Goals

- Unscoped queries answer in **≤30 seconds**.
- No reliance on a paid fallback — `ANTHROPIC_API_KEY` is being removed.
- Retrieval should surface multiple differing sources on a topic, not
  collapse to one "best" match, so `generate` can actually synthesize.

## Non-goals

- "Search this document" (scoped) queries — already fast, unaffected.
- Ingestion/chunking/embedding pipeline — unaffected.
- The SQL vectorless path — not implicated (0 SQL tables exist today).

## Design

### 1. Local embedding pre-filter before any LLM call

In `vectorless_pageindex.py`, before invoking `_pick_section` per tree, embed
the question once (reusing the already-loaded embedding model — no new
dependency) and compare it against each tree's own top-level summaries
(already computed and stored at ingestion) via cosine similarity. No LLM
call, no API round trip, near-instant. Keep only the top ~8 candidate trees;
skip the rest for this query entirely.

### 2. Batch the LLM relevance/section-pick call

Replace one `_pick_section` LLM call per tree with a single batched call
across the pre-filtered candidate set (~8 trees' summaries at once) —
mirroring the pattern `grade_documents` already uses for retrieved docs.
Ask the model to return **every** plausibly relevant tree/section, not just
one, matching the "collect multiple perspectives" goal. This drops LLM calls
at this stage from ~45+ down to 1 (plus at most one further small batched
call per matched tree that has children to descend into, bounded by the
existing `MAX_WALK_DEPTH`).

### 3. Drop the free-model fallback for this one call site

For the batched pageindex relevance call specifically, skip the OpenRouter
free-tier fallback on failure; default to keeping the candidate in play
("fail open toward inclusion") rather than waiting on a slow/unreliable free
model that has been observed padding heavily with hidden reasoning tokens.
Every other call site (final generation, `grade_documents`, guardrails, SQL
generation, page-index summarization at ingestion) keeps its existing
Groq → free OpenRouter chain unchanged — those are low-frequency (one call
per query), so the existing fallback is cheap insurance, not a burst risk.

### 4. Remove the Anthropic tier

Blank/remove `ANTHROPIC_API_KEY` in `.env`. No code change required —
`config.get_llm()` and `get_independent_judge_llm()` already degrade
gracefully when the key is absent; the eval-promotion pipeline
(`eval/promote_candidates.py`) already handles `get_independent_judge_llm()`
returning `None` by skipping promotion for that run.

### 5. Bound the retry loop's cost

In `graph.py`, on a `transform_query` retry, don't re-run the full vectorless
fan-out from scratch. Reuse the page-index/SQL results already fetched on the
first pass — rewording the question doesn't change which section an LLM
would pick the same way it changes vector-similarity results — and only
re-run the cheap vector-search leg with the rewritten query. Add a hard
per-query time ceiling as a safety net (independent of the above tuning) so
a worst case still can't exceed the 30-second budget; if exceeded, fall
through to `no_related_answer` rather than continuing to retry.

### 6. Loosen relevance retention for broad/definitional questions

Allow more than one page-index section per query to survive into `generate`
(not just one "best" per tree), and relax `grade_documents`'s bar slightly so
multiple partial-but-related sections can be kept and synthesized rather than
requiring a single dominant match. This is the piece that actually serves
"different authors, different definitions of the same topic, one synthesized
answer."

## Error handling

- Every new/changed LLM call site keeps this codebase's existing fail-open
  philosophy (see `guardrails.py`, `grade_documents`): a call failure never
  crashes the query, it degrades to a safe default (include the candidate,
  skip grading, etc.) and logs a `[ERROR]`/`[WARN]`-style print, matching
  existing conventions.
- The pre-filter (step 1) is pure local computation (embeddings + cosine
  similarity) — no network call, so no new failure mode to handle there.

## Testing plan

- Manual: re-run "What is RAG?" and "What are tokens?" unscoped in the Chat
  UI; confirm the answer returns in ≤30 seconds and visibly draws on more
  than one source document (check the Sources panel).
- Inspect `data/vector_store/telemetry.duckdb` after the run: confirm no
  `anthropic` provider rows, and confirm `pageindex_pick`-equivalent call
  volume dropped from ~45+ to a small handful.
- Confirm "search this document" (scoped) queries are unaffected — still
  single-document, still fast, unchanged behavior.

## Files touched

- `src/rag_learn/vectorless_pageindex.py` — pre-filter, batched pick,
  call-site fallback behavior, multi-section retention
- `src/rag_learn/graph.py` — retry loop reuse of first-pass fan-out results,
  hard cost/time ceiling, `grade_documents` leniency for broad questions
- `.env` — remove `ANTHROPIC_API_KEY`
