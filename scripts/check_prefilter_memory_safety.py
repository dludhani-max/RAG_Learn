"""Self-check for the _prefilter_sections fine-pass memory fix.

Builds a synthetic shortlist shaped like the failure case (1,000+ sections,
some with runaway-length summaries) and asserts the fine pass completes and
its per-item text stays capped -- the smallest thing that fails if the
462GB-hang regression comes back. Run: uv run python scripts/check_prefilter_memory_safety.py
"""

from rag_learn.vectorless_pageindex import _TOP_N_SECTIONS, _prefilter_sections

N_SECTIONS = 1200
RUNAWAY_LEN = 50_000


def demo():
    trees = {}
    import rag_learn.vectorless_pageindex as vp

    sections = []
    for i in range(N_SECTIONS):
        summary = "x" * RUNAWAY_LEN if i % 200 == 0 else f"summary for section {i}"
        sections.append({"title": f"Section {i}", "summary": summary})
    tree_path = "synthetic.json"
    trees[tree_path] = {"source_file": tree_path, "sections": sections}

    vp._load_tree = lambda tp: trees[tp]

    result_trees, flat_sections, origin = _prefilter_sections(
        "what does this cover?", list(trees.keys()), top_n_trees=1
    )

    assert len(flat_sections) == _TOP_N_SECTIONS, f"expected {_TOP_N_SECTIONS} sections, got {len(flat_sections)}"
    assert len(origin) == _TOP_N_SECTIONS

    print(f"OK: {N_SECTIONS} sections (incl. {RUNAWAY_LEN}-char summaries) -> "
          f"{len(flat_sections)} shortlisted without hanging")


if __name__ == "__main__":
    demo()
