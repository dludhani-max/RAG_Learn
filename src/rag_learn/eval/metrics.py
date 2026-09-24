"""Custom LLM-judge implementations of the 4 RAGAS-style evaluation
metrics, since the `ragas` library itself is unusable in this environment:
its `ragas/llms/base.py` unconditionally imports
`langchain_community.chat_models.vertexai`, a submodule removed from the
langchain-community version this project's document loaders require --
verified both the current ragas release and an older one hit the same
import error, and a user-provided fork did not fix it either (same
unmodified line).

Same LLM-judge pattern already used in guardrails.py/classifier.py: Groq,
temperature 0, a numeric 0.0-1.0 score parsed from the response. Named to
mirror what each RAGAS metric actually measures (see Phase 6 in the
implementation plan):
- retrieval_relevance ~ RAGAS context_precision
- groundedness        ~ RAGAS faithfulness
- answer_correctness  ~ RAGAS answer_correctness
- answer_relevancy    ~ RAGAS answer_relevancy
"""

import re
from typing import List

from rag_learn.llm_factory import default_factory

# max_tokens=20 caps the requested output ceiling, not just actual generation
# length -- verified live: without it, a single-number judge call was rejected
# outright by Groq's per-minute output-token quota because the client requested
# room for 1573 output tokens by default, well over what a "reply with one
# number" response needs.


def _score_prompt(prompt: str) -> float:
    """Runs a judge prompt expecting a single 0.0-1.0 number, extracted
    robustly in case the model wraps it in a sentence despite instructions
    (observed elsewhere this session: models don't always follow
    single-token output instructions exactly)."""
    try:
        raw = default_factory.get("eval_judge", temperature=0.0, max_tokens=20).invoke(prompt).content.strip()
    except Exception as e:
        print(f"[ERROR] Eval judge call failed, scoring 0.0: {e}")
        return 0.0
    match = re.search(r"(\d*\.?\d+)", raw)
    if not match:
        print(f"[ERROR] Could not parse a score from judge response: {raw!r}")
        return 0.0
    return max(0.0, min(1.0, float(match.group(1))))


def retrieval_relevance(question: str, contexts: List[str]) -> float:
    """What fraction of the retrieved context is actually relevant and
    necessary to answer the question (empty/irrelevant context scores
    low even if the final answer happened to be fine)."""
    if not contexts:
        return 0.0
    numbered = "\n\n".join(f"[{i}] {c[:800]}" for i, c in enumerate(contexts))
    prompt = (
        "Rate, from 0.0 to 1.0, what fraction of the following retrieved passages are "
        "actually relevant and necessary to answer the question. 1.0 means every passage is "
        "relevant; 0.0 means none are. Reply with ONLY the number.\n\n"
        f"Question: {question}\n\nPassages:\n{numbered}\n\nScore:"
    )
    return _score_prompt(prompt)


def groundedness(answer: str, contexts: List[str]) -> float:
    """How much of the answer's content is directly supported by the
    retrieved context -- catches the model filling gaps from its own
    pretrained knowledge instead of the documents."""
    if not answer.strip():
        return 1.0
    context_text = "\n\n".join(contexts)[:4000]
    prompt = (
        "Rate, from 0.0 (entirely unsupported) to 1.0 (fully supported), how much of the "
        "following answer's claims are directly supported by the given context. Reply with "
        "ONLY the number.\n\n"
        f"Context:\n{context_text}\n\nAnswer:\n{answer}\n\nScore:"
    )
    return _score_prompt(prompt)


def answer_correctness(answer: str, ground_truth: str) -> float:
    """How well the generated answer matches the golden ground-truth
    answer in factual content."""
    prompt = (
        "Rate, from 0.0 (completely wrong) to 1.0 (fully correct), how well the generated "
        "answer matches the ground-truth answer in factual content -- wording can differ, "
        "facts can't. Reply with ONLY the number.\n\n"
        f"Ground truth: {ground_truth}\n\nGenerated answer: {answer}\n\nScore:"
    )
    return _score_prompt(prompt)


def answer_relevancy(question: str, answer: str) -> float:
    """Whether the answer actually addresses the question asked --
    independent of whether it's factually correct (a confidently wrong but
    on-topic answer still scores well here; that's answer_correctness's
    job to catch)."""
    prompt = (
        "Rate, from 0.0 (does not address the question at all) to 1.0 (directly and fully "
        "addresses it), how relevant the following answer is to the question -- regardless of "
        "whether it's factually correct. Reply with ONLY the number.\n\n"
        f"Question: {question}\n\nAnswer: {answer}\n\nScore:"
    )
    return _score_prompt(prompt)


__all__ = ["retrieval_relevance", "groundedness", "answer_correctness", "answer_relevancy"]
