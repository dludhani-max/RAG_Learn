# RAG_Learn

A multi-format, agentic RAG application. Drop documents (text, Excel, PDF, images) into `data/`,
and query them through a Streamlit UI backed by LangChain/LangGraph/LangSmith, ChromaDB, and
vectorless retrieval paths (SQL for tabular data, page-index for structured documents).

Status: under active development. This README is updated as each build phase lands — see
`.claude/plans` (if present) for the full implementation plan.

## Setup

1. Install dependencies (uv-managed): `uv sync`
2. Copy `.env.example` to `.env` and fill in your keys:
   - `GROQ_API_KEY` — required. Used for generation and LLM-judge steps (grading, guardrails). Get one at https://console.groq.com/keys
   - `TAVILY_API_KEY` — required for the web-search fallback in the agentic retrieval flow. Get one at https://tavily.com
   - `LANGSMITH_API_KEY` — required for tracing and evaluation runs. Get one at https://smith.langchain.com
   - `OPENAI_API_KEY` / `GOOGLE_API_KEY` — optional, only needed if you switch providers later.
3. `.env` is gitignored — never commit real keys. `.env.example` stays tracked with placeholders only.

## Running

(To be filled in as the Streamlit app and CLI entry point are built.)
