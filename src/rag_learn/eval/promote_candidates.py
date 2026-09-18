"""Promotes highly-rated real Chat exchanges (see ratings.py) into
golden-dataset candidates. A real user rating a real answer 4-5 stars is a
strong signal -- arguably stronger than a synthetic question, since it
reflects an actual person judging whether they got a complete, correct
answer to something they actually asked. But per this project's existing
review discipline (golden_dataset.py's draft-then-review flow), nothing
merges into the real golden set without a human looking at it first --
this script's output is a pending-review file, not the golden set itself.

An independent judge (OpenRouter's free Nemotron model, via
config.get_independent_judge_llm -- deliberately not Groq, whatever
provider actually answered the original question) drafts a clean
ground_truth from the question + retrieved context, rather than just
promoting the original answer's own wording verbatim. This catches cases
where a user's high rating was generous despite a subtly imprecise
phrasing, and flags (via `verified: false`) cases where the retrieved
context doesn't actually support a confident answer at all.

Usage: uv run python3 -m rag_learn.eval.promote_candidates
"""

import json
import re
from pathlib import Path
from typing import Optional

from rag_learn import config, ratings

PENDING_PATH = Path(config.VECTOR_STORE_DIR) / "eval" / "rating_promoted_candidates.json"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


def _draft_ground_truth(judge, question: str, answer: str, contexts: list[str]) -> Optional[dict]:
    # Cap at 5 -- a highly-rated exchange could carry many sources, and this
    # is a one-shot judge call, not the retrieval path itself.
    context_block = "\n\n".join(contexts[:5]) or "(no retrieved context recorded for this exchange)"
    prompt = (
        "A user rated the following AI-generated answer highly (4 or 5 out of 5 stars). You are "
        "an independent reviewer, not the model that wrote this answer -- verify it against the "
        "retrieved context and produce a clean, well-written ground-truth answer suitable for an "
        "evaluation dataset. If the original answer is already accurate and well-written, your "
        "ground truth can closely match it; if it has any inaccuracy or missing detail relative "
        "to the context, correct that in your version.\n\n"
        'Respond with ONLY a JSON object: {"ground_truth": "...", "verified": true or false, '
        '"notes": "..."}. verified=false means the retrieved context doesn\'t actually support a '
        "confident answer to this question -- flag it for human attention rather than guessing.\n\n"
        f"Question: {question}\n\nRetrieved context:\n{context_block}\n\nOriginal answer:\n{answer}\n\nJSON:"
    )
    try:
        raw = _strip_think(judge.invoke(prompt).content)
        raw = _CODE_FENCE_RE.sub("", raw.strip())
        return json.loads(raw)
    except Exception as e:
        print(f"[PROMOTE] Judge call/parse failed: {e}")
        return None


def promote_candidates(min_rating: int = 4) -> list[dict]:
    judge = config.get_independent_judge_llm(temperature=0.0, max_tokens=500, purpose="rating_promotion_judge")
    if judge is None:
        print("[PROMOTE] No independent judge available (OPENROUTER_API_KEY not set) -- nothing promoted.")
        return []

    exchanges = ratings.list_promotable(min_rating=min_rating)
    if not exchanges:
        print("[PROMOTE] No promotable rated exchanges found.")
        return []

    pending = json.loads(PENDING_PATH.read_text()) if PENDING_PATH.exists() else []
    new_candidates = []

    for ex in exchanges:
        sources = json.loads(ex["sources_json"] or "[]")
        contexts = [s.get("content", "") for s in sources]
        print(f"[PROMOTE] Reviewing: {ex['question'][:70]!r} (user rated {ex['rating']}/5)")
        draft = _draft_ground_truth(judge, ex["question"], ex["answer"], contexts)
        # Mark promoted regardless of outcome -- a failed judge call
        # shouldn't be retried forever on every future run; it's still
        # visible in the ratings store for manual follow-up if needed.
        ratings.mark_promoted(ex["id"])
        if draft is None:
            continue
        candidate = {
            "question": ex["question"],
            "ground_truth": draft.get("ground_truth", ex["answer"]),
            "source": ex.get("target_document"),
            "original_answer": ex["answer"],
            "user_rating": ex["rating"],
            "user_comment": ex.get("rating_comment"),
            "judge_verified": draft.get("verified"),
            "judge_notes": draft.get("notes"),
            "exchange_id": ex["id"],
        }
        pending.append(candidate)
        new_candidates.append(candidate)
        print(f"  -> drafted (judge verified={draft.get('verified')})")

    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(pending, indent=2, default=str))
    print(f"\n{len(new_candidates)} new candidate(s) written to {PENDING_PATH}")
    print("Review this file by hand before merging any entries into golden_dataset_draft.json.")
    return new_candidates


if __name__ == "__main__":
    promote_candidates()
