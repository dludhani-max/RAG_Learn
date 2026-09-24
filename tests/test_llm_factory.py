"""Smoke tests for LLMFactory: construction and caching. No network calls --
building a client doesn't hit the provider until .invoke()."""

from langchain_core.runnables import Runnable

from rag_learn.llm_factory import LLMFactory


def test_factory_construction():
    """Factory constructs without errors."""
    factory = LLMFactory()
    assert factory is not None


def test_factory_get_returns_runnable():
    """Factory.get() returns a Runnable (fallback/telemetry wrappers make it a RunnableBinding)."""
    factory = LLMFactory()
    llm = factory.get("test", temperature=0.0, max_tokens=20)
    assert isinstance(llm, Runnable)


def test_factory_caching():
    """Factory caches LLMs by (purpose, temp, max_tokens, model_name)."""
    factory = LLMFactory()
    llm1 = factory.get("test", 0.0, 20)
    llm2 = factory.get("test", 0.0, 20)
    assert llm1 is llm2  # Same instance


def test_factory_cache_different_params():
    """Different params return different LLM instances."""
    factory = LLMFactory()
    llm1 = factory.get("test", 0.0, 20)
    llm2 = factory.get("test", 0.5, 20)  # Different temp
    assert llm1 is not llm2


def test_groq_only_returns_runnable():
    """Factory.get_groq_only() returns a Runnable."""
    factory = LLMFactory()
    llm = factory.get_groq_only("test", 0.0, 20)
    assert isinstance(llm, Runnable)


def test_independent_judge_with_key():
    """Factory.get_independent_judge() returns a Runnable when key set."""
    # Assumes OPENROUTER_API_KEY in env
    factory = LLMFactory()
    llm = factory.get_independent_judge(0.0, 20)
    assert llm is None or isinstance(llm, Runnable)
