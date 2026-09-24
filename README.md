# RAG_Learn

A multi-format, agentic RAG application. Drop documents (text, Excel, PDF, images) into `data/`,
and query them through a Streamlit UI backed by LangChain/LangGraph/LangSmith, ChromaDB, and
vectorless retrieval paths (SQL for tabular data, page-index for structured documents).

Status: under active development. This README is updated as each build phase lands — see
`.claude/plans` (if present) for the full implementation plan.

## The model stack, in plain terms

Three different jobs, three different kinds of model:

- **Chunking** (splitting a long document into smaller pieces): no AI at all, just a text-splitting
  rule (`RecursiveCharacterTextSplitter`) that cuts text into ~1750-character pieces with a little
  overlap so context isn't lost at the seams.
- **Embedding** (turning text into a "meaning fingerprint" so similar ideas can be matched):
  `Qwen/Qwen3-Embedding-0.6B` — a 0.6-billion-parameter model, ~1.5GB on first download. Runs
  **locally on your machine**, not through any API — free, no tokens, no internet call.
- **Retrieval** (finding the right content for a question) uses one of three paths depending on the
  document, only one of which needs the LLM:
  - Plain text/PDFs → **semantic search**: the embedding model above finds similar-meaning chunks,
    then a second local model (`BAAI/bge-reranker-v2-m3`, a ~568M-parameter cross-encoder, also
    free/local) double-checks which ones truly fit best.
  - Spreadsheets/tables → the LLM **writes a database query**, which then runs directly against the
    data (no embedding involved).
  - Long structured documents (books, guides) → a free local pre-filter (the embedding model above,
    scoring document- then section-level summaries) narrows the field, then one batched LLM call
    **picks the right section(s)** from that shortlist and reads them directly (no full-corpus
    embedding involved).
- **Everything that requires actually understanding, deciding, writing, or judging language** goes
  through Groq — two models, picked per call site by cost/latency tradeoff, both overridable via env
  vars:
  - `qwen/qwen3.8-27b` (`RAG_GROQ_MODEL_NAME`, ~27B parameters) — the default for generation and
    most judging: checking a question is safe to answer, writing SQL (tabular data), judging
    whether what was retrieved is actually relevant, rewriting the question and retrying if not,
    writing the final answer, and checking that answer isn't fabricated or inappropriate before
    it's shown to you.
  - `openai/gpt-oss-20b` (`RAG_GROQ_BATCH_MODEL_NAME`) — used only for the page-index batched section-pick call, which fires in a
    short burst per unscoped query; a smaller/faster model here keeps that burst from becoming the
    latency bottleneck, and it's called through a no-fallback client (`default_factory.get_groq_only`) since a
    bursty volume of calls would otherwise saturate a fallback tier just as badly as Groq itself.
  This Groq usage is the only part of the pipeline that costs API tokens or calls out over the
  internet. See "Guardrails" and "Vectorless retrieval" below for the full detail on each step.

**Intended scale:** this is built and tuned for **personal or small-team** document sets (tens to
low hundreds of files) — not an enterprise document store. The unscoped "search everything" query
path fans out across every SQL table and page-index tree in the corpus (see "Vectorless retrieval"
below), so its cost scales with *how many separate documents* you've ingested, not with total corpus
size in general. At personal scale this stays well under the 30-second target; a corpus with
hundreds of separately-routed documents would need re-tuning (e.g. `_TOP_N_TREES`/`_TOP_N_SECTIONS`
in `vectorless_pageindex.py`) before it'd still feel fast.

## Why it's built this way, in plain terms

A few design choices that aren't obvious just from reading the code, explained simply:

- **Search happens in two steps, not one.** The first pass is fast but rough — it just compares
  "does this look similar" across everything. The second pass is slower but much more careful — it
  looks closely at only the top candidates from the first pass and re-checks each one properly
  against the actual question. Doing the careful check on everything would be too slow; doing only
  the rough check would miss things. Two steps gets both speed and accuracy.

- **The app double-checks its own search results before answering.** If the first search comes back
  empty or off-topic, the app doesn't just give up (or worse, guess) — it rewrites the question in a
  different way and searches again, up to a couple of extra tries. Only after that does it accept
  "nothing relevant found" as the final answer, and it says so honestly instead of making something
  up.

- **Different document types are searched in different ways, on purpose.** Cramming everything
  through the same search method loses information: a spreadsheet's rows and columns don't make
  sense as a chunk of text, and a whole book doesn't fit into one small search result. So tables get
  their own database lookup, long structured documents get their own outline-based search, and
  everything else uses the general text search. Each document is searched the way that actually
  fits its shape.

- **A new idea was tested before being added, and it turned out not to help.** Adding a second,
  different search method (matching exact words instead of just meaning) was a real, live-tested
  question — not just adopted because it's a common technique. The test showed the app's existing
  two-step search already handled tricky cases (like acronyms) well once both steps were properly
  compared, so the extra complexity wasn't added. Proving something doesn't help is still a useful
  result, not a wasted one.

## Quick Start

**Prerequisites:** Python 3.14+, [`uv`](https://docs.astral.sh/uv/), a free [Groq API key](https://console.groq.com/keys).
On macOS, also `brew install tesseract` (needed for scanned PDFs and images — see step 1 below).

```bash
brew install tesseract          # macOS only; see your OS's package manager otherwise
uv sync                         # installs everything, including Python itself if needed
cp .env.example .env            # then open .env and paste in your GROQ_API_KEY
uv run streamlit run streamlit_app.py
```

Open the URL it prints (usually `http://localhost:8501`). You'll land on the **Home** page.

**First run, what to expect:**
1. Drop a few files into `data/` (PDFs, text files, images — whatever you want to ask questions about).
2. Go to the **Ingest** page and click **Run Sync**. The very first sync downloads the embedding
   model (~1.5GB) and a reranker model, so it can take a few minutes — after that, re-syncing is
   fast (only new/changed files are processed).
3. Go to **Chat** and ask a question. That's it.

That's the whole loop: **Ingest** to index your documents, **Chat** to ask about them. Everything
below this section is deeper reference material (architecture, configuration, guardrails, the
evaluation framework) — worth reading once you're past first setup, not required to get started.

## Setup

1. Install the Tesseract OCR binary (used for image ingestion — `pytesseract` is just a wrapper around it, it does not bundle the binary): `brew install tesseract` on macOS. Without this, image files in `data/` will fail to load (each failure is caught per-file, so it won't crash ingestion, but images will silently contribute no content).
2. Install Python dependencies (uv-managed): `uv sync`
3. Copy `.env.example` to `.env` and fill in your keys:
   - `GROQ_API_KEY` — **required**, the only key the app checks for on startup. Used for generation and every LLM-judge step (grading, guardrails, classification). Get one at https://console.groq.com/keys
   - `LANGSMITH_API_KEY` — optional. Without it the app runs fine; with it, every query gets traced (retrieve/grade/generate step-by-step, viewable in the LangSmith dashboard) and the evaluation framework (see below) can log experiments. Get one at https://smith.langchain.com. Tracing is scoped to whichever key is in your own `.env` — each LangSmith account has its own monthly trace quota, and tracing failures (e.g. quota exceeded) are logged but never block a query from answering; if you see `LangSmithRateLimitError` in the logs, that's just your account's trace quota, not an app bug.
   - `OPENAI_API_KEY` / `GOOGLE_API_KEY` — optional, only needed if you switch providers later.
4. `.env` is gitignored — never commit real keys. `.env.example` stays tracked with placeholders only.

## Configuration

All pipeline settings (paths, embedding model, chunk size, retrieval/eval knobs) live in
`src/rag_learn/config.py` and can be overridden via environment variables (see that file for the
full list, e.g. `RAG_CHUNK_SIZE`, `RAG_TOP_K`). Nothing else in the codebase should hardcode these
values — change them here.

Notable defaults:
- Embedding model: `Qwen/Qwen3-Embedding-0.6B` (free, local, ~1.5GB on first download) — chosen
  over smaller/faster alternatives for better retrieval quality; swappable via `RAG_EMBEDDING_MODEL`
  if ingestion throughput ever becomes the bottleneck (e.g. many concurrent users uploading large
  document sets).
- Chunk size/overlap: 1750/300 characters, sized to that model's larger context window.

## Running

Three entry points, depending on what you want:

```
uv run streamlit run streamlit_app.py   # interactive web UI (Home/Ingest/Chat) -- normal usage
uv run python3 app.py                    # ingest only: sync data/, no query, no LLM calls
uv run rag-learn                         # full smoke test: sync data/ + one query through the graph
```

`app.py` and `rag-learn` both run an incremental **sync** of every supported file under `data/`
(PDF, TXT, CSV, Excel, Word, JSON, and OCR'd images), routing each one to vector embedding, a
DuckDB SQL table, or a page-index tree depending on its classification (see "Document
classification" below). `rag-learn` additionally asks one sample question through the full
Corrective/Adaptive RAG graph and prints the answer with its sources.

You can also run just the sync step directly: `uv run python3 -m rag_learn.sync`

### How sync works

Re-running ingestion is cheap and safe — a manifest (`data/vector_store/manifest.json`, gitignored)
tracks each file's content hash, so unchanged files are skipped entirely rather than re-embedded.
On each run:
- **New file** → chunked, embedded, added.
- **Unchanged file** → skipped (no re-embedding cost).
- **Modified file** (content changed) → its old chunks are removed and it's re-ingested fresh.
- **Deleted file** → its chunks are removed from the vector store.
- **Renamed/moved file** (same content, new path) → detected automatically, just updates the
  manifest — no re-embedding.
- **A new file that looks like a version of an existing one** (e.g. `resume_v2.pdf` next to
  `resume.pdf`, matched by filename similarity) → the old version is only ever auto-replaced when
  there's a clear, unambiguous signal that the new one is more recent (a date in the filename, or
  failing that, file modification time). Otherwise both are kept and the pair is logged to
  `data/vector_store/pending_review.json` for you to review manually — nothing is silently deleted
  on a guess.
- Changing pipeline settings that affect chunk shape (`RAG_EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`,
  `RAG_CHUNK_OVERLAP`) or the loader logic itself invalidates the whole cache and triggers a full
  re-sync, so old and new chunk formats never silently mix in the same collection.

### Agentic query flow (LangGraph)

Once the vector store is populated, ask questions through the full Corrective/Adaptive RAG graph:

```
uv run python3 -m rag_learn.graph
```

Flow: `retrieve` (fetches `RAG_RETRIEVE_CANDIDATES`, default 20, by vector similarity, then
reranks down to `RAG_TOP_K` with a local cross-encoder, `BAAI/bge-reranker-v2-m3` — more accurate
than vector similarity alone since it scores the query and chunk jointly rather than comparing
separately-computed embeddings) → `grade_documents` (an LLM call judges which of those are
actually relevant) → if enough are relevant, `generate`; if not and retries remain,
`transform_query` rewrites the question and loops back to `retrieve`; once retries are exhausted,
**`no_related_answer`** returns a fixed "No related answers found." response. Answers include a
deduplicated source list (local file/page), each with the actual retrieved text it's based on, not
just a filename/page pointer. Retry count is capped by `RAG_MAX_RETRIES` (default 2) so a bad query
can't loop forever.

There is deliberately no web-search fallback: an earlier version of this graph fell back to Tavily
web search when local retrieval came up empty, but per an explicit decision an answer must only
ever come from what's actually indexed (vector or vectorless RAG) — never the open web — so "found
nothing locally" is a terminal response, not a trigger to look further out.

Grading and query-rewrite calls use `reasoning_effort="none"` on the Groq model to skip its
internal `<think>` reasoning output for these short structured tasks — cheaper and faster, since
nothing reads that reasoning for a yes/no grade or a one-line rewrite.

### Semantic Q&A cache

The very first node in the graph is `check_cache`: it embeds the incoming question and checks it
against a shared (not per-session) cache of past question→answer pairs. If a close-enough match is
found (cosine similarity ≥ `RAG_CACHE_SIMILARITY_THRESHOLD`, default 0.95 — deliberately strict,
since this reuses an answer wholesale rather than just influencing retrieval), the cached answer is
returned immediately with zero retrieval/reranking/grading/generation cost. Paraphrased questions
hit the cache too (verified: "What programming languages does Deepak know?" and "Which coding
languages is Deepak familiar with?" matched at similarity 0.960). A successful `generate` always
writes its answer back to the cache via a `store_cache` node.

A rejection from either output guardrail (groundedness or toxicity, see below) is **never** cached,
even though `output_guardrail` still returns a fixed safe message the same shape as a real answer —
a guardrail judgment can be wrong (a false-negative groundedness rejection on a genuinely correct
answer), and caching that mistake would make it permanent: every future similar question would keep
replaying the same stale rejection instead of getting a fresh judgment. Only an answer that actually
passed both guardrails is worth reusing.

The cache is automatically wiped by `sync()` whenever the document set actually changes (add,
update, remove, or an auto-replaced version — a rename alone doesn't, since the content is
unchanged) — a stale cached answer being served silently would be worse than a cache miss, so
invalidation is deliberately coarse (clear everything) rather than trying to track which cached
answers depended on which source documents.

If `LANGSMITH_API_KEY` is set in `.env`, every node in the graph is automatically traced — check
the LangSmith dashboard (project `rag-learn`) to see the retrieve/grade/generate (and any
retry/web-search) sequence for a given query. Tracing is a no-op with no visible effect if the key
isn't set.

### Diagram extraction

While loading a PDF's natively-extracted pages (not OCR-fallback pages — a scanned page's
"embedded image" is just the whole page render, already covered by OCR), each embedded raster
image is saved to `data/vector_store/extracted_images/` if it's at least 150px in both dimensions
(filters out icons/bullets/decorative elements — verified against this corpus: real diagrams run
1000px+, decorative elements were 200x62 / 500x300) and not a byte-identical repeat of an image
already seen elsewhere in the same document (filters out repeated cover banners/logos — verified:
`QA AI.pdf`'s title banner appeared on 3 separate pages). A standalone image file (`.jpg`/`.png`)
is treated the same way, referencing its own path. Saved diagram paths travel with their source
chunk through the vector store (JSON-encoded in Chroma metadata, since Chroma only accepts scalar
values) and are attached per-source-citation in the graph's `generate` node output, so an answer's
source list can point to the specific diagram(s) on the page it drew from.

Known limitation: this only removes *exact* duplicate images. A document that reuses a
template-style section-header banner with different text per section (same size, same visual
style, different pixels) will still surface each as a separate "diagram" — distinguishing
decorative template chrome from genuine content diagrams would need perceptual-similarity
matching, which isn't implemented.

### Document classification (vector vs. vectorless routing)

Every ingested file is tagged with a `routing` value before chunking, so a future retrieval path
(Phase 3b, not yet built) can serve tabular data and long structured documents differently instead
of forcing everything through embedding search:
- `vector` — flat/narrative content (résumés, cover letters, OCR'd images without tabular
  structure, unsectioned PDFs). The only route actually queried today.
- `vectorless_sql` — tabular files (`.csv`, `.xlsx`, `.xls`) by extension alone, plus images whose
  OCR'd text looks like a grid/table.
- `vectorless_pageindex` — PDFs/Word docs with real structure: either a PDF bookmark/outline, or
  (when no outline exists) enough heading-like lines in the extracted text — e.g. a short 6-page
  guide with numbered "Prompt 1"–"Prompt 6" sections still routes here despite its length, since
  routing follows structure, not page count.

Images are classified with a hybrid approach to control cost: a free local heuristic (delimiter
density, average line length) resolves clear-cut cases; only genuinely ambiguous OCR text falls
back to a single Groq classification call. `routing` is stored alongside each chunk in the vector
store, defaulting to `vector` for anything the classifier doesn't recognize. Check
`[SYNC] Routing decisions this run: {...}` in `sync()`'s output to see the counts per route.

### Guardrails

Every query passes through guardrail checks at both ends of the graph, in `src/rag_learn/guardrails.py`:

- **Input safety** (`input_safety_guardrail`, the very first node — runs before cache lookup or
  retrieval): an LLM classifies the question as SAFE or UNSAFE (prompt injection, jailbreak
  attempts, requests for harmful content). An UNSAFE question is refused immediately
  ("I can't help with that request.") without ever touching the cache, retrieval, or generation —
  and the refusal itself is never cached. An ordinary question that's simply unrelated to the
  corpus is still SAFE; that's a retrieval problem, not a safety one (see below).
- **Corpus relevance**: not a separate check — the existing `grade_documents` retry loop already
  determines this from actual retrieval results (more accurate than guessing from the question
  text alone), and its terminal "nothing relevant after retries" case now returns
  `no_related_answer`'s fixed "No related answers found." response.
- **PII redaction**, on both the incoming question and the outgoing answer/source content:
  deterministic, regex-based (no extra LLM call) detection of email, phone, credit card, IP, MAC
  address, and URL, replaced with `[REDACTED_TYPE]` placeholders. Applied uniformly regardless of
  the corpus — note this means an answer drawing on a personal document (e.g. a résumé) will come
  back with its own contact info redacted too, by explicit choice. Free-text PII like a person's
  name is out of scope: reliable name detection needs NER, which isn't a deterministic/no-LLM-call
  operation, so it isn't caught.
- **Output groundedness** (`output_guardrail`, after `generate`): a second LLM call checks that
  every claim in the answer is actually supported by the retrieved context, catching cases where
  the generation model fills a gap from its own pretrained knowledge instead of saying the context
  doesn't cover it. An ungrounded answer is replaced with a fixed message and its sources are
  discarded — nothing partially-verified is ever returned.
- **Output toxicity**, same node: a second classification catches harassment/hate-speech/harmful
  content in the generated answer before it's returned or cached.

All guardrail LLM checks fail open (default to SAFE/GROUNDED on an API error) so a transient Groq
outage degrades to "no extra check ran" rather than blocking every query outright.

### Vectorless retrieval (SQL + page-index)

`vectorless_sql`- and `vectorless_pageindex`-routed files skip embedding entirely -- they're ingested
into their own retrieval mechanism instead, and `sync()`'s add/update/delete flow manages each one's
lifecycle (dropping a DuckDB table or deleting a tree file, the same way it deletes vector chunks):

- **SQL path** (`vectorless_sql.py`): each tabular file becomes its own DuckDB table
  (`data/vector_store/vectorless.duckdb`). At query time, a Groq call translates the question into a
  single **read-only SELECT** against that table's actual schema (a regex guardrail rejects anything
  containing INSERT/UPDATE/DELETE/DROP/etc., even disguised behind a stacked `;` statement) and the
  result rows come back as a markdown table. The generation prompt asks for every column relevant to
  the question, not just the one literally named -- a query for "engineers earning over 100k" selects
  `name, department, salary`, not just `name`, so the answer is verifiable from the table alone
  without needing to see the SQL that produced it.
- **Page-index path** (`vectorless_pageindex.py`): each structured document gets a one-time tree of
  sections built at ingestion -- a PDF's own bookmark/outline when it has one (exact page ranges), or
  a heading-line split (the same heuristic that routed it here in the first place) otherwise. Each
  section gets a short LLM summary; a chapter split into many children instead gets a summary built
  from their titles, capped at 500 characters -- verified live, one chapter split into 755 children
  produced an 11,544-character uncapped summary, and several such oversized summaries in one batched
  prompt blew past Groq's per-minute input-token limit outright. At query time, one LLM call picks
  the single most relevant section from the summaries (or none, if nothing fits) and its full text
  is returned -- no embedding similarity involved.

Both are exposed through `run_query(question, target_document=...)`: passing a filename (via the
Streamlit Chat page's "Scope" picker, or directly) scopes retrieval to that one document (matched
against a SQL table, a page-index tree, or -- for a vector-routed document -- a `source_file`
metadata filter on the normal vector search) instead of searching everything.

Leaving `target_document` unset ("search everything") **fans out across all three retrieval paths**
-- vector search, every SQL table, every page-index tree -- and merges the results, letting
`grade_documents` filter the combined set for actual relevance. This was originally deferred (per
the plan's own recommendation, to avoid the added LLM-call cost of checking every vectorless
document on every query) until live testing showed why it mattered: most of a typical corpus ends
up page-index- or SQL-routed rather than vector-routed, so an unscoped question about that content
always returned "no related answer" without this -- a much worse default than the extra cost of
checking. The added cost scales with how many SQL tables + page-index trees exist, not with corpus
size in general, which stays reasonable at personal/small-team scale.

### Evaluation framework

```
uv run python3 -m rag_learn.eval.golden_dataset   # draft Q&A pairs -> data/vector_store/eval/golden_dataset_draft.json
# --- review/edit the draft file by hand here ---
uv run python3 -c "from rag_learn.eval.golden_dataset import push_to_langsmith; push_to_langsmith()"
uv run python3 -m rag_learn.eval.run_eval          # scores every golden question, logs a LangSmith experiment
```

Four metrics, computed per question and logged to a LangSmith experiment against the
`rag-learn-golden` dataset (viewable in the LangSmith UI): `retrieval_relevance`, `groundedness`,
`answer_correctness`, `answer_relevancy`. **Not implemented with the `ragas` library** -- it's
unusable in this environment (`ragas/llms/base.py` unconditionally imports
`langchain_community.chat_models.vertexai`, a submodule removed from the langchain-community
version this project's document loaders require; verified against both the current release and an
older one, and a separately-provided fork did not fix it either). Implemented instead as custom
Groq LLM-judge functions (`eval/metrics.py`), the same pattern used throughout `guardrails.py` and
`classifier.py`. LangSmith itself (dataset storage + experiment tracking) is unaffected and works
normally either way -- `ragas` was only ever the scoring engine, not the dashboard.

Golden dataset generation is LLM-assisted but explicitly a draft: `golden_dataset.py` writes
candidate Q&A pairs to a local JSON file for human review/editing before `push_to_langsmith()`
uploads them -- auto-generated ground truth is a starting point, not authoritative. For a document
that's itself already structured as Q&A (e.g. an interview-question guide), extracting real
question/answer pairs directly from its text is both cheaper and more trustworthy than having an
LLM invent new ones from an excerpt.

`run_eval.py`'s target function scopes each question to the specific document it was drafted from
(via `target_document`) -- necessary because a question drawn from a `vectorless_pageindex`-routed
document is only answerable through that path; an unscoped query only ever searches the vector
store, which by design never receives page-index-routed content.

A real run surfaced a genuine finding, not just noise: several correct answers still scored 0.0 on
`retrieval_relevance`, because the page-index path returns a whole chapter-level section (thousands
of characters covering many subtopics) for a single-paragraph question -- the answer was right, but
most of the retrieved context was irrelevant filler. That's a legitimate signal about page-index
granularity, exactly what this framework exists to surface.

### Rating-based golden dataset promotion

A second path into the golden dataset, alongside `golden_dataset.py`'s LLM-drafted candidates: real
Chat answers a user actually rated highly.

```
uv run python3 -m rag_learn.eval.promote_candidates   # -> data/vector_store/eval/rating_promoted_candidates.json
# --- review this file by hand, same as golden_dataset_draft.json ---
```

Every Chat answer is persisted (question, answer, sources) the moment it's generated
(`ratings.record_exchange`), and the star-rating widget under each answer attaches a 1-5 score to
it. `promote_candidates.py` pulls every exchange rated 4-5 stars and not yet promoted
(`ratings.list_promotable`), and for each one, an **independent judge LLM** -- OpenRouter's free
`nvidia/nemotron-3-super-120b-a12b:free`, deliberately a different provider than Groq (which
answers every real question) -- re-verifies the answer against its retrieved context and drafts a
clean `ground_truth`, rather than promoting the user's approved wording verbatim. This catches a
generous rating on a subtly imprecise answer, and flags (`verified: false`) any case where the
retrieved context doesn't actually support a confident answer at all.

A real user rating a real answer is arguably a stronger signal than a synthetic question -- but per
this project's review discipline, nothing merges into the actual golden set without a human looking
at it first. `promote_candidates.py`'s output is a separate pending-review file, never merged
directly into `golden_dataset_draft.json`.

### Streamlit UI

```
uv run streamlit run streamlit_app.py
```

Four pages (standard Streamlit multipage app -- `streamlit_app.py` is Home, `pages/` holds the rest):
- **Home**: status dashboard -- API key check, vector/SQL/page-index counts, indexed documents grouped by routing.
- **Ingest**: a data-directory field and a "Run Sync" button, thin orchestration over `sync()` (no ingestion logic duplicated in the UI). Also surfaces the pending-review queue (see "How sync works" above) with a dismiss action.
- **Chat**: the same `run_query()`/graph used everywhere else, with a sidebar "Scope" picker built from whatever's actually indexed -- pick a specific document to invoke Phase 3b's `target_document` scoping, or leave it on "Search everything" (which now genuinely searches every retrieval path, not just vector -- see below). Sources render with their full retrieved content and any attached diagrams, not just a filename/page citation.
- **Insights**: recent chat traces and evaluation experiment scores, pulled live via the LangSmith API and rendered as native tables -- see "Evaluation framework" below for why this isn't a literal embed of the LangSmith site. Requires `LANGSMITH_API_KEY`; the page explains itself and skips gracefully if it's not set.

Expensive resources (embedding model, reranker, compiled graph) are cached once per server process via `st.cache_resource`, so they only load on the first chat message, not on every rerun.

Any backend failure (an exhausted Groq quota, a transient API error) degrades to a plain-language chat message -- never a raw traceback in the UI. Verified live: every LLM call in the graph is wrapped with a fallback, and a real exhausted-quota condition was reproduced and confirmed to degrade gracefully rather than crash the page.
