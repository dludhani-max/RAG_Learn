"""Single source of truth for env loading and pipeline settings.

Every other module (data_loader, embedding, vectorstore, search, graph,
guardrails, eval, and the Streamlit app) should read settings from here
instead of hardcoding defaults, so there's one place to change behavior.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.getenv("RAG_DATA_DIR", str(PROJECT_ROOT / "data"))
VECTOR_STORE_DIR = os.getenv("RAG_VECTOR_STORE_DIR", str(Path(DATA_DIR) / "vector_store"))

# "learn_documents" (the notebook-era collection) holds 384-dim vectors from
# all-MiniLM-L6-v2. Qwen3-Embedding-0.6B produces 1024-dim vectors, which is
# a hard incompatibility for a ChromaDB collection (fixed embedding
# dimension) -- so this pipeline uses a new collection name and re-ingests
# from scratch rather than mixing dimensions or overwriting the old data.
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "learn_documents_v2")

# --- Embedding & chunking ------------------------------------------------
# Qwen3-Embedding-0.6B over the notebook's all-MiniLM-L6-v2: better MTEB
# retrieval quality, still free/local/Apache-2.0. Its larger context window
# also removes the ~256-token ceiling that was already capping MiniLM-based
# chunk sizes near 1000 chars, so chunk_size is raised accordingly below.
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1750"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "300"))

# --- Retrieval -----------------------------------------------------------
# Retrieve-then-rerank: fetch a wider candidate set by vector similarity
# (cheap, coarse), then a cross-encoder scores each (query, chunk) pair
# directly for a more accurate relevance ranking (slower per-pair but only
# run over RETRIEVE_CANDIDATES items, not the whole collection) and keep
# just the top TOP_K. A cross-encoder sees the query and chunk together,
# unlike embedding similarity which compares them independently -- that
# consistently ranks true relevance better.
TOP_K = int(os.getenv("RAG_TOP_K", "10"))
RETRIEVE_CANDIDATES = int(os.getenv("RAG_RETRIEVE_CANDIDATES", "20"))
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))

# Caps how many of the final top_k slots a single source document can fill.
# Verified live: with no cap, plain score-sorting let one document's chunks
# fill every slot even in "search everything" mode, silently starving other
# genuinely relevant documents out of the answer -- working against this
# app's whole point of helping a user learn from everything relevant in the
# corpus, not just whichever single document scored marginally higher.
# 3 is an untuned default -- never compared against 2/4/5 with eval data.
MAX_PER_SOURCE = int(os.getenv("RAG_MAX_PER_SOURCE", "3"))

# --- Semantic Q&A cache ----------------------------------------------------
# A high threshold: this is an exact-answer-reuse cache, not a retrieval
# threshold, so a false-positive "hit" for a subtly different question
# would silently return a wrong answer -- err strict.
CACHE_COLLECTION_NAME = os.getenv("RAG_CACHE_COLLECTION_NAME", "qa_cache")
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("RAG_CACHE_SIMILARITY_THRESHOLD", "0.95"))

# --- Generation / judge LLM ----------------------------------------------
GROQ_MODEL_NAME = os.getenv("RAG_GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Small/fast model for page-index's bursty batched section pick only (see
# llm_factory.get_groq_only). groq/compound-mini was retired by Groq
# (404 model_not_found, verified live 2026-09-24).
GROQ_BATCH_MODEL_NAME = os.getenv("RAG_GROQ_BATCH_MODEL_NAME", "openai/gpt-oss-20b")

# --- Anthropic fallback LLM (Tier 2 router) -------------------------------
# Every LLM call in this app goes through llm_factory.py, which falls back
# here when Groq fails (rate limit, exhausted daily quota, outage) --
# verified live this session: Groq's quota is scoped per-model, and this
# app's single GROQ_MODEL_NAME is shared by all 9 call sites, so one bulk
# ingestion run can exhaust the day's budget for every other call too.
# Optional: with no ANTHROPIC_API_KEY set, the factory just returns the Groq
# client with no fallback attached (today's original behavior), so this
# degrades gracefully rather than requiring the new credential.
ANTHROPIC_MODEL_NAME = os.getenv("RAG_ANTHROPIC_MODEL_NAME", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- OpenRouter free-model tier (sits between Groq and Anthropic) ---------
# The factory's full chain: Groq (primary) -> this small chain of free
# OpenRouter models -> Anthropic (paid, last resort). Each free model is a
# separate rate-limit bucket, so this absorbs a Groq quota exhaustion
# without needing to fall all the way to paid Anthropic. OpenRouter is
# OpenAI-API-compatible, so it's constructed via ChatOpenAI with a custom
# base_url, not a dedicated LangChain package.
#
# Checked live against openrouter.ai/collections/free-models on 2026-09-08
# -- that catalog visibly rotates over time (nothing matched what older
# training data would suggest), so these are env-overridable defaults, not
# hardcoded assumptions that should be trusted indefinitely.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAMES = [
    m.strip()
    for m in os.getenv(
        "RAG_OPENROUTER_MODEL_NAMES",
        "nvidia/nemotron-3-super-120b-a12b:free,liquid/lfm-2.5-2.6b:free",
    ).split(",")
    if m.strip()
]

# --- LangSmith tracing -----------------------------------------------------
# Set both the legacy LANGCHAIN_* names and the current LANGSMITH_* names
# defensively, since different langsmith/langchain versions have looked for
# either family of env vars.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-learn")

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
else:
    # .env may set LANGSMITH_TRACING=true as a placeholder before a key is
    # filled in -- without a key that just makes every LangChain call spam
    # 401 auth warnings trying to report traces nobody can see. Force
    # tracing off explicitly rather than relying on whatever raw value
    # load_dotenv() happened to pull in.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def missing_required_keys() -> list[str]:
    """Required keys that aren't set, so the UI/CLI can surface config problems early."""
    required = {"GROQ_API_KEY": GROQ_API_KEY}
    return [name for name, value in required.items() if not value]
