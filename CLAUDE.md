# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG_Learn is an early-stage learning project for building a Retrieval-Augmented Generation (RAG) pipeline in Python. It currently implements PDF document loading and text embedding; vector storage and search are not yet implemented.

## Environment & Commands

- Python >= 3.14, managed with `uv` (see `pyproject.toml` / `uv.lock`)
- Install dependencies: `uv sync`
- Run the example pipeline: `uv run python app.py`
- Entry point script: `rag-learn` -> `rag_learn:main` (currently just prints a placeholder message)
- No test suite, linter, or formatter is configured yet.

## Architecture

- `src/rag_learn/data_loader.py` — `load_all_documents(data_dir)` loads PDF files (via `PyPDFLoader`) recursively from a directory into LangChain `Document` objects. Loader imports for TXT/CSV/Excel/Word/JSON are present but not yet wired into the function.
- `src/rag_learn/embedding.py` — `EmbeddingPipeline` chunks documents with `RecursiveCharacterTextSplitter` (default 1000 chars / 200 overlap) and embeds them with `sentence-transformers` (default model `all-MiniLM-L6-v2`).
- `src/rag_learn/vectorstore.py`, `src/rag_learn/search.py` — empty stubs; vector storage (chromadb/faiss-cpu are dependencies) and retrieval/search are not yet implemented.
- `app.py` (repo root) — driver script that loads documents from `data/`, chunks them, and embeds them; used to manually exercise the pipeline end-to-end so far.
- `data/` — working data directory (`PDFLearn/`, `text_files/`, `vector_store/`).
- `notebook/` — exploratory Jupyter notebooks (`document.ipynb`, `pdf_loader.ipynb`) for document/PDF loading, separate from the `src/` package.

## Known Issues

- `EmbeddingPipeline.embed_chunks` in `src/rag_learn/embedding.py` references an undefined `text` variable in a debug print (should be `texts`).
