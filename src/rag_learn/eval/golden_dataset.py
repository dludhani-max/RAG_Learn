"""Golden dataset generation for Phase 6 evaluation.

LLM-assisted draft generation of question/ground-truth pairs from the
actual indexed corpus. Per the plan: auto-generated ground truth is a
starting draft, not authoritative -- a human reviews/edits the draft file
before it becomes the real golden set pushed to LangSmith.

Flow:
1. generate_candidates() -- writes a draft JSON file for human review.
2. A human reviews/edits that file directly (fix wrong answers, drop bad
   questions, adjust wording).
3. push_to_langsmith() -- uploads the REVIEWED file to a LangSmith dataset.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_learn import config
from rag_learn.llm_factory import default_factory
from rag_learn.data_loader import load_document
from rag_learn.sync import list_indexed_documents

DRAFT_PATH = Path(config.VECTOR_STORE_DIR) / "eval" / "golden_dataset_draft.json"
DATASET_NAME = "rag-learn-golden"
QUESTIONS_PER_DOC = 2

# Some temperature (0.3, unlike the deterministic judge calls elsewhere) --
# these are draft questions for a human to edit, not a pass/fail
# classification, so a bit of variety is fine. The factory suppresses reasoning
# output (model-family-aware) and max_tokens=600 caps the ceiling -- without
# either, this hit two real failures verified live: (1) the model's reasoning
# preceded the JSON, breaking every single parse; (2) Groq rejected the call
# outright (1491-2014 requested output tokens vs. a 1000/min budget) since
# nothing capped the requested ceiling.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()


def _generate_qa_for_text(text: str, source: str, n: int) -> List[Dict[str, str]]:
    prompt = (
        f"Given the following document excerpt, write {n} question-and-answer pairs a user "
        "might genuinely ask about it. Each answer must be fully supported by the excerpt -- "
        "do not invent facts not present in it. Output ONLY a JSON array of objects with keys "
        '"question" and "ground_truth", nothing else, no markdown fences.\n\n'
        f"Excerpt:\n{text[:4000]}"
    )
    try:
        raw = default_factory.get("golden_dataset_generation", temperature=0.3, max_tokens=600).invoke(prompt).content
    except Exception as e:
        print(f"[ERROR] Golden Q&A generation failed for {source}: {e}")
        return []
    raw = _strip_code_fence(raw)
    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[ERROR] Could not parse Q&A JSON for {source}: {raw[:200]!r}")
        return []
    return [
        {"question": p["question"], "ground_truth": p["ground_truth"], "source": source}
        for p in pairs
        if isinstance(p, dict) and p.get("question") and p.get("ground_truth")
    ]


def generate_candidates(
    data_dir: Optional[str] = None, questions_per_doc: int = QUESTIONS_PER_DOC
) -> List[Dict[str, Any]]:
    """Draft Q&A pairs across every currently-indexed document (spanning
    whichever routing types -- vector, SQL, page-index -- actually exist in
    the corpus). Writes the draft to DRAFT_PATH and returns it; does NOT
    touch LangSmith -- see push_to_langsmith() for that, after review."""
    data_path = Path(data_dir or config.DATA_DIR)
    indexed = list_indexed_documents()
    all_names = sorted({name for names in indexed.values() for name in names})

    candidates: List[Dict[str, Any]] = []
    for name in all_names:
        matches = list(data_path.glob(f"**/{name}"))
        if not matches:
            print(f"[EVAL] Skipping '{name}' -- no longer found under {data_path}")
            continue
        path = matches[0]
        docs = load_document(path)
        if not docs:
            continue
        text = "\n".join(d.page_content for d in docs)
        pairs = _generate_qa_for_text(text, name, questions_per_doc)
        candidates.extend(pairs)
        print(f"[EVAL] Drafted {len(pairs)} Q&A pairs from {name}")

    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_text(json.dumps(candidates, indent=2))
    print(f"[EVAL] Wrote {len(candidates)} draft Q&A pairs to {DRAFT_PATH}")
    print("[EVAL] REVIEW this file before calling push_to_langsmith() -- these are unverified drafts.")
    return candidates


def push_to_langsmith(reviewed_path: Path = DRAFT_PATH, dataset_name: str = DATASET_NAME) -> int:
    """Upload a human-reviewed draft file to a LangSmith dataset. Creates
    the dataset if it doesn't exist yet. Returns the number of examples
    uploaded."""
    from langsmith import Client

    examples = json.loads(Path(reviewed_path).read_text())
    if not examples:
        print("[EVAL] No examples to upload.")
        return 0

    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name, description="RAG_Learn golden Q&A set (human-reviewed)."
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": e["question"]},
                "outputs": {"ground_truth": e["ground_truth"]},
                "metadata": {"source": e.get("source", "")},
            }
            for e in examples
        ],
    )
    print(f"[EVAL] Uploaded {len(examples)} examples to LangSmith dataset '{dataset_name}'.")
    return len(examples)


if __name__ == "__main__":
    generate_candidates()
