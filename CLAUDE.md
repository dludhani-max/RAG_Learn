# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG_Learn is a multi-format, agentic RAG application. Documents (text, Excel, PDF, images) dropped
into `data/` are ingested via classification-aware routing into one of three retrieval mechanisms
(vector, SQL, or page-index), then queried through a LangGraph Corrective/Adaptive RAG flow with
guardrails on both ends, exposed via a Streamlit UI. See `README.md` for full usage docs and
`~/.claude/plans/okay-now-you-know-mossy-galaxy.md` for the phased implementation plan this project
was built against (not checked into this repo).

## Environment & Commands

- Python >= 3.14, managed with `uv` (see `pyproject.toml` / `uv.lock` -- the only dependency source
  of truth; there is no `requirements.txt`)
- Install dependencies: `uv sync`
- Ingest documents only: `uv run python3 app.py`
- Full smoke test (ingest + one query through the whole graph): `uv run rag-learn`
- Web UI: `uv run streamlit run streamlit_app.py`
- No test suite, linter, or formatter is configured yet.

## Architecture

- `src/rag_learn/data_loader.py` — per-file loaders for PDF (per-page native/OCR fallback, table
  extraction, diagram extraction), TXT, CSV, Excel, Word, JSON, images (OCR). `load_document(path)`
  dispatches by extension; `load_all_documents(data_dir)` is a bulk alternative not used by the
  actual ingestion path (that's `sync.py`).
- `src/rag_learn/classifier.py` — tags every ingested file with a `routing` value (`vector` /
  `vectorless_sql` / `vectorless_pageindex`) before chunking.
- `src/rag_learn/embedding.py` — chunks documents and embeds them (`Qwen/Qwen3-Embedding-0.6B`).
- `src/rag_learn/vectorstore.py` — ChromaDB wrapper for the `vector`-routed collection.
- `src/rag_learn/vectorless_sql.py` — DuckDB text-to-SQL path for `vectorless_sql`-routed (tabular)
  files. Uses short-lived per-call connections, not a persistent one (DuckDB takes an exclusive file
  lock per connection).
- `src/rag_learn/vectorless_pageindex.py` — hierarchical section-tree path for
  `vectorless_pageindex`-routed (long structured) files; LLM navigates the tree at query time.
- `src/rag_learn/search.py` — `Retriever`, vector similarity search with an optional `source_file`
  scope filter.
- `src/rag_learn/reranker.py` — cross-encoder reranking of vector search candidates.
- `src/rag_learn/cache.py` — semantic Q&A cache (`QACache`), a separate Chroma collection.
- `src/rag_learn/sync.py` — the actual production ingestion path: incremental add/update/remove/
  rename detection via a manifest, dispatches each file to the right routing's ingestion mechanism.
- `src/rag_learn/graph.py` — the LangGraph orchestration: `input_safety_guardrail -> check_cache ->
  retrieve -> grade_documents -> [generate | transform_query retry loop | no_related_answer] ->
  output_guardrail -> store_cache`. `run_query(question, target_document=...)` is the main entry
  point. No web-search fallback exists (removed by design -- answers only ever come from indexed
  content).
- `src/rag_learn/guardrails.py` — input safety classification, PII redaction (input + output),
  output groundedness and toxicity checks.
- `streamlit_app.py` + `pages/` — the web UI (Home/Ingest/Chat), a standard Streamlit multipage app.
- `app.py` (repo root) — ingest-only CLI script (`sync()`, no query).
- `src/rag_learn/__init__.py`'s `main()` — the `rag-learn` CLI entry point: ingest + one query
  through the full graph, a smoke test that the whole pipeline works end-to-end.
- `data/` — working data directory (`PDFLearn/`, `text_files/`, `vector_store/` — the last one holds
  all persisted state: ChromaDB, the sync manifest, DuckDB tables, page-index tree JSON files; it's
  gitignored).
- `notebook/` — early exploratory Jupyter notebooks, superseded by the `src/` package.

## Known Issues

None currently tracked. If you find one, add it here with the file/symptom, not just "known issues
exist" -- a stale or vague entry here is worse than no entry.
