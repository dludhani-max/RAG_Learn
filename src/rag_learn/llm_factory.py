"""Centralized LLM construction: the one place every module gets an LLM from.

Replaces the per-module `_get_*_llm()` lazy singletons -- the factory caches
internally by (purpose, temperature, max_tokens, model_name), so callers just
ask for what they need every time:

    from rag_learn.llm_factory import default_factory

    llm = default_factory.get("classification", temperature=0.0, max_tokens=20)
    answer = llm.invoke(prompt).content

- `get()`                   -- Groq -> free OpenRouter models -> Anthropic fallback chain
- `get_groq_only()`         -- Groq with no fallback and no retries, for bursty call sites
- `get_independent_judge()` -- OpenRouter Nemotron, never Groq; None without a key

Keys default to the env-loaded values in config.py; pass them explicitly to
override (e.g. in tests).
"""

from langchain_core.runnables import Runnable

from rag_learn import config, telemetry

# Verified live: both current free OpenRouter models spend real completion
# tokens on hidden reasoning before answering (60-86 tokens for a trivial
# "reply with one word" prompt) -- the same failure mode handled for Groq's
# gpt-oss models in _build_llm_kwargs() below. Unlike Groq, there's no single
# fix here: one model (Nemotron) accepts OpenRouter's unified
# `reasoning: {enabled: false}` and drops to 0 reasoning tokens, but the other
# (LFM-2.5) rejects that exact request outright ("Reasoning is mandatory for
# this endpoint and cannot be disabled."). Rather than special-case per model,
# just pad max_tokens generously for every OpenRouter call -- these are free
# models, so the wasted tokens cost nothing, and this stays correct
# automatically if the free-model roster changes later.
_OPENROUTER_REASONING_PADDING = 300

# Nemotron (120B), not the other free tier (LFM-2.5, 2.6B) -- too small to
# trust as a judge.
_INDEPENDENT_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def _tag(runnable, provider: str):
    # Every client is tagged "provider:<name>" so telemetry.py can attribute
    # usage to whichever one actually served a call without guessing from its
    # class name -- load-bearing for OpenRouter specifically, since it's
    # constructed via the same ChatOpenAI class real OpenAI would use.
    return runnable.with_config({"tags": [f"provider:{provider}"]})


class LLMFactory:
    """Centralized LLM construction with Groq -> OpenRouter -> Anthropic fallback."""

    def __init__(
        self,
        groq_key: "str | None" = None,
        openrouter_key: "str | None" = None,
        anthropic_key: "str | None" = None,
    ):
        """Construct factory. Keys default to env vars (via config) if None."""
        self.groq_key = groq_key or config.GROQ_API_KEY
        self.openrouter_key = openrouter_key or config.OPENROUTER_API_KEY
        self.anthropic_key = anthropic_key or config.ANTHROPIC_API_KEY
        self._cache: dict[tuple, Runnable] = {}

    @staticmethod
    def _build_llm_kwargs(
        temperature: float = 0.0, max_tokens: "int | None" = None, model_name: str = config.GROQ_MODEL_NAME
    ) -> dict:
        """Model-family-aware kwargs for constructing a ChatGroq client, so
        every call site gets a valid reasoning-suppression setting regardless
        of which model GROQ_MODEL_NAME points at, instead of each one
        hardcoding a value that only works for one model family.

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

    def get(
        self,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: "int | None" = None,
        model_name: "str | None" = None,
    ) -> Runnable:
        """Get an LLM wrapped in a fallback chain (LangChain's native
        `.with_fallbacks()`, which retries the next entry on any exception
        from the previous one -- rate limits included):

            Groq `model_name` (default config.GROQ_MODEL_NAME) -> free
            OpenRouter models (if an OpenRouter key is set) -> Anthropic
            (paid last resort, if an Anthropic key is set)

        Each stage past Groq is optional. `purpose` is a short label (e.g.
        "classification", "generation") used to break down telemetry by call
        site. Cached by (purpose, temperature, max_tokens, model_name).
        """
        key = ("get", purpose, temperature, max_tokens, model_name)
        if key in self._cache:
            return self._cache[key]

        from langchain_groq import ChatGroq

        groq_kwargs = self._build_llm_kwargs(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name or config.GROQ_MODEL_NAME
        )
        chain = [_tag(ChatGroq(**groq_kwargs, api_key=self.groq_key), "groq")]

        if self.openrouter_key:
            from langchain_openai import ChatOpenAI

            for or_model in config.OPENROUTER_MODEL_NAMES:
                chain.append(
                    _tag(
                        ChatOpenAI(
                            model=or_model,
                            temperature=temperature,
                            max_tokens=(max_tokens or 1024) + _OPENROUTER_REASONING_PADDING,
                            api_key=self.openrouter_key,
                            base_url=config.OPENROUTER_BASE_URL,
                        ),
                        "openrouter",
                    )
                )

        if self.anthropic_key:
            from langchain_anthropic import ChatAnthropic

            # Anthropic's API requires max_tokens explicitly, unlike Groq --
            # callers that leave it unset (e.g. graph.py's generation LLM)
            # still need a real cap here so the fallback doesn't fail outright.
            chain.append(
                _tag(
                    ChatAnthropic(
                        model=config.ANTHROPIC_MODEL_NAME,
                        temperature=temperature,
                        max_tokens=max_tokens or 1024,
                        api_key=self.anthropic_key,
                    ),
                    "anthropic",
                )
            )

        llm = chain[0].with_fallbacks(chain[1:]) if len(chain) > 1 else chain[0]
        llm = llm.with_config({"callbacks": [telemetry.TelemetryCallback(purpose=purpose)]})
        self._cache[key] = llm
        return llm

    def get_groq_only(
        self,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: "int | None" = None,
        model_name: "str | None" = None,
    ) -> Runnable:
        """Groq-only, NO fallback chain -- for a call site that fires many
        times per query in a short burst (see vectorless_pageindex's batched
        relevance pick). Verified live: a bursty volume of calls saturates
        the free OpenRouter fallback tier just as badly as Groq itself, and
        waiting 10+ seconds per call on a free model that ignores "answer
        with one number" instructions costs more than it's worth for a cheap
        relevance check. Callers must handle a failure themselves (fail open
        toward inclusion, not exclusion).

        max_retries=0: LangChain's default ChatGroq retries a rate-limited
        call with backoff before raising -- verified live, this turned a
        burst of 18 rate-limited calls into 343 seconds of stacked "please
        wait 15-18s" delays instead of failing in under a second each.
        """
        key = ("groq_only", purpose, temperature, max_tokens, model_name)
        if key in self._cache:
            return self._cache[key]

        from langchain_groq import ChatGroq

        llm = _tag(
            ChatGroq(
                **self._build_llm_kwargs(
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name or config.GROQ_BATCH_MODEL_NAME
                ),
                api_key=self.groq_key,
                max_retries=0,
            ),
            "groq",
        )
        llm = llm.with_config({"callbacks": [telemetry.TelemetryCallback(purpose=purpose)]})
        self._cache[key] = llm
        return llm

    def get_independent_judge(
        self,
        temperature: float = 0.0,
        max_tokens: "int | None" = None,
        purpose: str = "independent_judge",
    ) -> "Runnable | None":
        """A judge deliberately NOT sharing a provider with get()'s primary
        (Groq) -- for grading a candidate independently of whatever model
        produced it (see eval/promote_candidates.py). Uses OpenRouter's free
        Nemotron model directly, not the Groq-primary fallback chain, which
        would route through Groq first and defeat the point.

        Returns None if no OpenRouter key is set, rather than silently
        falling back to Groq -- callers must handle the None case explicitly
        (e.g. skip promotion, tell the user why).
        """
        if not self.openrouter_key:
            return None

        from langchain_openai import ChatOpenAI

        llm = _tag(
            ChatOpenAI(
                model=_INDEPENDENT_JUDGE_MODEL,
                temperature=temperature,
                max_tokens=(max_tokens or 1024) + _OPENROUTER_REASONING_PADDING,
                api_key=self.openrouter_key,
                base_url=config.OPENROUTER_BASE_URL,
            ),
            "openrouter",
        )
        # primary_provider="openrouter" (not the TelemetryCallback default of
        # "groq") -- OpenRouter IS the intended provider here, so labeling it
        # is_fallback=True would misrepresent the Insights fallback-rate metric.
        return llm.with_config(
            {"callbacks": [telemetry.TelemetryCallback(purpose=purpose, primary_provider="openrouter")]}
        )


default_factory = LLMFactory()
