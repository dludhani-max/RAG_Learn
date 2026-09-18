"""One-off check for whether the Corrective-RAG retry loop (rewrite the
question and retry retrieval on a bad first pass) is actually worth its
extra latency -- not a permanent part of the app, just answers the question
before deciding whether to change anything.

Runs a small subset of the golden dataset UNSCOPED (target_document=None,
so it exercises the full "search everything" fan-out where the retry loop
actually gets exercised) once with retries on (default) and once with
retries forced off (RAG_MAX_RETRIES=0), and compares latency + scores.

Kept to 5 questions to hold down both wall-clock time and LLM token cost --
this is a directional check, not a full eval run.

Usage: uv run python3 scripts/check_retry_value.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "data" / "vector_store" / "eval" / "golden_dataset_draft.json"

# A small, mixed subset (index into the golden dataset) -- not the full 20,
# to keep this check cheap. Indices chosen for topic spread, not difficulty.
SUBSET_INDICES = [0, 3, 8, 10, 19]


def _run_one(question: str, ground_truth: str, max_retries: str) -> dict:
    """Runs one question through the real graph in a fresh subprocess (so
    RAG_MAX_RETRIES, read once at import time in config.py, takes effect
    cleanly per run) and returns latency + quality scores."""
    code = (
        "import time, json;"
        "from rag_learn.graph import run_query;"
        "from rag_learn.eval import metrics;"
        f"q = {question!r};"
        f"gt = {ground_truth!r};"
        "t0 = time.time();"
        "result = run_query(q, target_document=None);"
        "elapsed = time.time() - t0;"
        "answer = result['generation'];"
        "contexts = [s.get('content', '') for s in result.get('sources', [])];"
        "scores = {"
        "  'groundedness': metrics.groundedness(answer, contexts),"
        "  'answer_correctness': metrics.answer_correctness(answer, gt),"
        "};"
        "print(json.dumps({'elapsed': elapsed, 'answer': answer, 'scores': scores}))"
    )
    env = {**os.environ, "RAG_MAX_RETRIES": max_retries}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        print(f"    [ERROR] subprocess failed: {proc.stderr[-500:]}")
        return {"elapsed": None, "answer": None, "scores": {}}
    # Last line is the JSON result; earlier lines are the app's own [INFO]/print noise.
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def main():
    entries = json.loads(GOLDEN_PATH.read_text())
    subset = [entries[i] for i in SUBSET_INDICES]

    rows = []
    for i, entry in enumerate(subset):
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        print(f"[{i + 1}/{len(subset)}] {question}")

        print("  retries ON (default)...")
        on = _run_one(question, ground_truth, max_retries="2")
        print(f"    {on['elapsed']:.1f}s  scores={on['scores']}" if on["elapsed"] else "    failed")

        print("  retries OFF...")
        off = _run_one(question, ground_truth, max_retries="0")
        print(f"    {off['elapsed']:.1f}s  scores={off['scores']}" if off["elapsed"] else "    failed")

        rows.append({"question": question, "on": on, "off": off})

    print("\n=== Summary ===")
    print(f"{'Question':<55} {'ON (s)':>8} {'OFF (s)':>8} {'ON correctness':>15} {'OFF correctness':>16}")
    for r in rows:
        on, off = r["on"], r["off"]
        on_t = f"{on['elapsed']:.1f}" if on["elapsed"] else "ERR"
        off_t = f"{off['elapsed']:.1f}" if off["elapsed"] else "ERR"
        on_c = on["scores"].get("answer_correctness", "-")
        off_c = off["scores"].get("answer_correctness", "-")
        print(f"{r['question'][:53]:<55} {on_t:>8} {off_t:>8} {str(on_c):>15} {str(off_c):>16}")


if __name__ == "__main__":
    main()
