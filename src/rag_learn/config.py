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

# --- Semantic Q&A cache ----------------------------------------------------
# A high threshold: this is an exact-answer-reuse cache, not a retrieval
# threshold, so a false-positive "hit" for a subtly different question
# would silently return a wrong answer -- err strict.
CACHE_COLLECTION_NAME = os.getenv("RAG_CACHE_COLLECTION_NAME", "qa_cache")
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("RAG_CACHE_SIMILARITY_THRESHOLD", "0.95"))

# --- Generation / judge LLM ----------------------------------------------
GROQ_MODEL_NAME = os.getenv("RAG_GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Anthropic fallback LLM (Tier 2 router) -------------------------------
# Every LLM call in this app goes through get_llm() below, which falls back
# here when Groq fails (rate limit, exhausted daily quota, outage) --
# verified live this session: Groq's quota is scoped per-model, and this
# app's single GROQ_MODEL_NAME is shared by all 9 call sites, so one bulk
# ingestion run can exhaust the day's budget for every other call too.
# Optional: with no ANTHROPIC_API_KEY set, get_llm() just returns the Groq
# client with no fallback attached (today's original behavior), so this
# degrades gracefully rather than requiring the new credential.
ANTHROPIC_MODEL_NAME = os.getenv("RAG_ANTHROPIC_MODEL_NAME", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- OpenRouter free-model tier (sits between Groq and Anthropic) ---------
# get_llm()'s full chain: Groq (primary) -> this small chain of free
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


def llm_kwargs(temperature: float = 0.0, max_tokens: "int | None" = None, model_name: str = GROQ_MODEL_NAME) -> dict:
    """Model-family-aware kwargs for constructing a ChatGroq client, so every
    LLM call site in the app (7 of them, across graph.py, guardrails.py,
    classifier.py, vectorless_sql.py, vectorless_pageindex.py, and eval/)
    gets a valid reasoning-suppression setting regardless of which model
    GROQ_MODEL_NAME points at, instead of each one hardcoding a value that
    only works for one model family.

    "none" is a Qwen-specific reasoning_effort value -- verified against
    Groq's own docs: gpt-oss models reject it outright (they only accept
    low/medium/high) and need a different mechanism entirely
    (`model_kwargs={"include_reasoning": False}`) to get a clean
    final-answer-only response the way Qwen's "none" does. Groq's
    agentic "compound" models reject reasoning_effort entirely (400
    "not supported with this model") and already emit zero reasoning
    tokens by default, so they get neither kwarg.
    """
    kwargs: dict = {"model_name": model_name, "temperature": temperature}
    if model_name.startswith("groq/compound"):
        pass
    elif model_name.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"
        kwargs["model_kwargs"] = {"include_reasoning": False}
        # Verified live: even at reasoning_effort="low" with include_reasoning
        # disabled, gpt-oss still spends real completion tokens on internal
        # reasoning before writing any visible answer (observed 12
        # reasoning_tokens against a max_tokens=20 cap, hitting finish_reason
        # "length" with empty content -- unlike Qwen's "none", which emits
        # zero reasoning tokens). Pad the caller's budget so that overhead
        # doesn't silently eat the whole response for short-answer call
        # sites tuned against Qwen's true zero-reasoning behavior.
        if max_tokens is not None:
            max_tokens += 150
    else:
        kwargs["reasoning_effort"] = "none"
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def get_llm(temperature: float = 0.0, max_tokens: "int | None" = None, purpose: str = "unspecified"):
    """The single construction point for every LLM this app uses -- every
    call site should call this instead of building a provider client
    directly. Returns Groq wrapped in a fallback chain (via LangChain's
    native `.with_fallbacks()`, which retries the next entry on any
    exception from the previous one -- rate limits included):

        Groq (primary) -> free OpenRouter models (OPENROUTER_MODEL_NAMES,
        if OPENROUTER_API_KEY is set) -> Anthropic (paid last resort, if
        ANTHROPIC_API_KEY is set)

    Each stage is optional -- with no OpenRouter/Anthropic keys set, this
    degrades to Groq-only (today's original behavior). Every client is
    tagged "provider:<name>" (`.with_config({"tags": [...]})`) so
    telemetry.py can attribute usage to whichever one actually served a
    call without having to guess from its class name -- load-bearing for
    OpenRouter specifically, since it's constructed via the same
    ChatOpenAI class real OpenAI would use. A telemetry callback recording
    every actual call to the local usage store is attached last. `purpose`
    is a short label (e.g. "classification", "generation") used to break
    down telemetry by call site.
    """
    from langchain_groq import ChatGroq

    from rag_learn import telemetry

    def _tag(runnable, provider: str):
        return runnable.with_config({"tags": [f"provider:{provider}"]})

    chain = [_tag(ChatGroq(**llm_kwargs(temperature=temperature, max_tokens=max_tokens)), "groq")]

    if OPENROUTER_API_KEY:
        from langchain_openai import ChatOpenAI

        # Verified live: both current free models spend real completion
        # tokens on hidden reasoning before answering (60-86 tokens for a
        # trivial "reply with one word" prompt) -- the same failure mode
        # already fixed for Groq's gpt-oss models in llm_kwargs() above.
        # Unlike Groq, there's no single fix here: one model (Nemotron)
        # accepts OpenRouter's unified `reasoning: {enabled: false}` and
        # drops to 0 reasoning tokens, but the other (LFM-2.5) rejects that
        # exact request outright ("Reasoning is mandatory for this endpoint
        # and cannot be disabled."). Rather than special-case per model,
        # just pad max_tokens generously for every OpenRouter call -- these
        # are free models, so the wasted tokens cost nothing, and this stays
        # correct automatically if the free-model roster changes later.
        _OPENROUTER_REASONING_PADDING = 300
        for model_name in OPENROUTER_MODEL_NAMES:
            chain.append(
                _tag(
                    ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=(max_tokens or 1024) + _OPENROUTER_REASONING_PADDING,
                        api_key=OPENROUTER_API_KEY,
                        base_url=OPENROUTER_BASE_URL,
                    ),
                    "openrouter",
                )
            )

    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic

        # Anthropic's API requires max_tokens explicitly, unlike Groq --
        # callers that leave it unset (e.g. graph.py's generation LLM) still
        # need a real cap here so the fallback doesn't fail outright.
        chain.append(
            _tag(
                ChatAnthropic(
                    model=ANTHROPIC_MODEL_NAME,
                    temperature=temperature,
                    max_tokens=max_tokens or 1024,
                    api_key=ANTHROPIC_API_KEY,
                ),
                "anthropic",
            )
        )

    llm = chain[0].with_fallbacks(chain[1:]) if len(chain) > 1 else chain[0]
    return llm.with_config({"callbacks": [telemetry.TelemetryCallback(purpose=purpose)]})


def get_llm_groq_only(
    temperature: float = 0.0,
    max_tokens: "int | None" = None,
    purpose: str = "unspecified",
    model_name: str = "groq/compound-mini",
):
    """Groq-only construction with NO fallback chain -- for a call site that
    fires many times per query in a short burst (see vectorless_pageindex's
    batched relevance pick). Verified live: a bursty volume of calls
    saturates the free OpenRouter fallback tier just as badly as Groq
    itself, and waiting 10+ seconds per call on a free model that ignores
    "answer with one number" instructions costs more than it's worth for a
    cheap relevance check. Callers at this call site must handle a failure
    themselves (fail open toward inclusion, not exclusion) rather than
    trusting a slow fallback to save them.

    max_retries=0: LangChain's default ChatGroq retries a rate-limited call
    with backoff before raising -- verified live, this turned a burst of
    18 rate-limited calls into 343 seconds of stacked "please wait 15-18s"
    delays instead of failing in under a second each. This call site's
    whole design already assumes a failure means "skip and move on", so
    the built-in retry only fights that."""
    from langchain_groq import ChatGroq

    from rag_learn import telemetry

    llm = ChatGroq(
        **llm_kwargs(temperature=temperature, max_tokens=max_tokens, model_name=model_name), max_retries=0
    ).with_config({"tags": ["provider:groq"]})
    return llm.with_config({"callbacks": [telemetry.TelemetryCallback(purpose=purpose)]})


def get_independent_judge_llm(temperature: float = 0.0, max_tokens: "int | None" = None, purpose: str = "independent_judge"):
    """A judge LLM deliberately NOT sharing a provider with get_llm()'s
    primary (Groq) -- for use where grading/validating a candidate
    independently of whatever model actually produced it matters (see
    eval/promote_candidates.py: an outside call cross-checks a
    highly-rated Chat answer before it becomes a golden-dataset candidate,
    rather than trusting the same model's own output as its own ground
    truth). Uses OpenRouter's free Nemotron model directly (not the
    Groq-primary fallback chain in get_llm() -- that would route through
    Groq first, defeating the point) -- a real second opinion at no cost,
    since the app's paid Anthropic tier was deliberately removed. Returns
    None if OPENROUTER_API_KEY isn't set, rather than silently falling back
    to Groq -- that would defeat the point of asking for an outside
    opinion, so callers must handle the None case explicitly (e.g. skip
    promotion, tell the user why) rather than treating this the way
    get_llm()'s optional tiers degrade.
    """
    if not OPENROUTER_API_KEY:
        return None
    from langchain_openai import ChatOpenAI

    from rag_learn import telemetry

    # Nemotron (120B), not the other free tier (LFM-2.5, 2.6B) -- too small
    # to trust as a judge. Same reasoning-padding need as get_llm()'s
    # OpenRouter stage: Nemotron accepts `reasoning: {enabled: false}` and
    # drops to 0 reasoning tokens, but pad anyway for safety since these
    # are free tokens.
    _INDEPENDENT_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
    llm = ChatOpenAI(
        model=_INDEPENDENT_JUDGE_MODEL,
        temperature=temperature,
        max_tokens=(max_tokens or 1024) + 300,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    ).with_config({"tags": ["provider:openrouter"]})
    # primary_provider="openrouter" here (not the TelemetryCallback default
    # of "groq") -- this call was never a fallback from anything, OpenRouter
    # IS the intended provider for this purpose, and mislabeling it
    # is_fallback=True would misrepresent the fallback-rate metric on the
    # Insights page.
    return llm.with_config(
        {"callbacks": [telemetry.TelemetryCallback(purpose=purpose, primary_provider="openrouter")]}
    )
