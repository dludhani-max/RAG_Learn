# RAG_Learn Domain Model

## Architecture

### LLMFactory

**Location:** `src/rag_learn/llm_factory.py`

Centralized LLM construction with Groq → OpenRouter → Anthropic fallback chain. Every module that needs an LLM should call `llm_factory.default_factory.get()` instead of maintaining its own singleton.

**Interface:**
- `get(purpose, temperature, max_tokens, model_name)` → Primary LLM with fallback
- `get_groq_only(...)` → Groq-only, no fallback, fast-fail (for high-frequency calls)
- `get_independent_judge(...)` → OpenRouter Nemotron judge (eval cross-check); `None` without an OpenRouter key

**Caching:** Internal cache keyed by `(purpose, temperature, max_tokens, model_name)`.

**Why:** Eliminates the scattered `_get_*_llm()` singletons, exposes one testable seam, makes provider changes 1-file edits.
