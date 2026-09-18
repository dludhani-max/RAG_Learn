"""Second half of the retry-loop check: the first pass (check_retry_value.py)
found the loop never fired on clean, well-covered questions -- cost nothing,
but didn't prove it helps either. This script uses deliberately harder
questions (vague phrasing, indirect wording) to see the retry loop actually
engage, and reports whether it changed anything.

No ground_truth scoring here (these questions are hand-written, not from
the golden set) -- this is about observing retry_count and whether the
final answer differs, not measuring correctness. Kept to 3 questions to
hold down cost.

Usage: uv run python3 scripts/check_retry_trigger.py
"""

import json
from pathlib import Path

from rag_learn.graph import run_query

ROOT = Path(__file__).resolve().parent.parent

# Deliberately vague/indirectly-worded questions, unscoped, meant to stress
# the first retrieval pass into missing and needing a rewrite.
QUESTIONS = [
    "What's the thing that controls how random or safe a model's word choices are?",
    "Why does a model sometimes just make stuff up?",
    "How do you decide which word comes next when there's more than one good option?",
]


def main():
    for i, question in enumerate(QUESTIONS):
        print(f"[{i + 1}/{len(QUESTIONS)}] {question}")
        result = run_query(question, target_document=None)
        print(f"  retry_count: {result.get('retry_count')}")
        print(f"  final question used: {result.get('question')}")
        print(f"  answer: {result.get('generation', '')[:300]}")
        print(f"  sources: {[s.get('source') for s in result.get('sources', [])]}")
        print()


if __name__ == "__main__":
    main()
