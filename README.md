# RAG_Learn

A multi-format, agentic RAG application. Drop documents (text, Excel, PDF, images) into `data/`,
and query them through a Streamlit UI backed by LangChain/LangGraph/LangSmith, ChromaDB, and
vectorless retrieval paths (SQL for tabular data, page-index for structured documents).

Status: under active development. This README is updated as each build phase lands — see
`.claude/plans` (if present) for the full implementation plan.

## Setup

1. Install the Tesseract OCR binary (used for image ingestion — `pytesseract` is just a wrapper around it, it does not bundle the binary): `brew install tesseract` on macOS. Without this, image files in `data/` will fail to load (each failure is caught per-file, so it won't crash ingestion, but images will silently contribute no content).
2. Install Python dependencies (uv-managed): `uv sync`
3. Copy `.env.example` to `.env` and fill in your keys:
   - `GROQ_API_KEY` — required. Used for generation and LLM-judge steps (grading, guardrails). Get one at https://console.groq.com/keys
   - `TAVILY_API_KEY` — required for the web-search fallback in the agentic retrieval flow. Get one at https://tavily.com
   - `LANGSMITH_API_KEY` — required for tracing and evaluation runs. Get one at https://smith.langchain.com
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

The core pipeline (sync -> retrieval, no agentic orchestration yet) can be smoke-tested end-to-end:

```
uv run python3 app.py
```

This runs an incremental **sync** of every supported file under `data/` (PDF, TXT, CSV, Excel,
Word, JSON, and OCR'd images) against the ChromaDB collection at `data/vector_store`, then runs
one sample retrieval query, printing the top matches with their similarity scores.

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
`web_search` (Tavily) fills in with live web results before generating. Answers include a
deduplicated source list (local file/page or web URL). Retry count is capped by `RAG_MAX_RETRIES`
(default 2) so a bad query can't loop forever.

Grading and query-rewrite calls use `reasoning_effort="none"` on the Groq model to skip its
internal `<think>` reasoning output for these short structured tasks — cheaper and faster, since
nothing reads that reasoning for a yes/no grade or a one-line rewrite.

If `LANGSMITH_API_KEY` is set in `.env`, every node in the graph is automatically traced — check
the LangSmith dashboard (project `rag-learn`) to see the retrieve/grade/generate (and any
retry/web-search) sequence for a given query. Tracing is a no-op with no visible effect if the key
isn't set.

The Streamlit UI is not wired up yet (later phase).
