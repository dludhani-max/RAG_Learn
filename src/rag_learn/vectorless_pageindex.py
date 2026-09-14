"""Vectorless page-index retrieval path: long structured documents
(routing == "vectorless_pageindex") get a one-time hierarchical tree index
built at ingestion (headings/sections + a short LLM summary per section),
then queried by having an LLM navigate the tree top-down at query time --
closer to how a person flips to the right chapter than to embedding
similarity, and avoids vector RAG's chunk-boundary information loss.

See Phase 3b.ii in the implementation plan for the full design/rationale.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
from langchain_core.documents import Document

from rag_learn import config
from rag_learn.classifier import ROUTING_PAGEINDEX, HEADING_LINE_RE, SENTENCE_END_RE

TREES_DIR = Path(config.VECTOR_STORE_DIR) / "pageindex"

# Bounds the query-time tree walk so a bad/ambiguous question can't spiral
# into unbounded LLM calls -- matches the plan's "bounded by tree depth so
# it stays cheap."
MAX_WALK_DEPTH = 3

# Bounds how many _summarize()/_pick_section() calls run concurrently.
# These are network round-trips, not CPU work, so a small thread pool cuts
# wall-clock time substantially (verified live: a 231-section PDF took
# minutes of one-call-at-a-time waiting) -- kept low rather than unbounded
# since Groq's per-minute token limit is an account-wide ceiling, not a
# per-connection one; too much concurrency just trades slow success for
# faster 429s.
_LLM_CONCURRENCY = 4

_pageindex_llm = None
_pageindex_picker_llm = None


def _get_pageindex_llm():
    # Used for _summarize(): asked for 1-2 sentences, so needs more room
    # than the picker below.
    global _pageindex_llm
    if _pageindex_llm is None:
        _pageindex_llm = config.get_llm(temperature=0.0, max_tokens=150, purpose="pageindex_summarize")
    return _pageindex_llm


def _get_pageindex_picker_llm():
    # Used for _pick_section(): the answer is just a section number (or
    # "none"). Verified live: leaving max_tokens unset let ChatGroq default
    # to ~2048, and Groq's output-tokens-per-minute limit is enforced
    # against the *declared* max_tokens, not actual usage -- an uncapped
    # request got rejected outright (429) even though the real answer is a
    # couple of characters.
    global _pageindex_picker_llm
    if _pageindex_picker_llm is None:
        _pageindex_picker_llm = config.get_llm(temperature=0.0, max_tokens=20, purpose="pageindex_pick")
    return _pageindex_picker_llm


_embedder = None

_TOP_N_CANDIDATES = 8


def _get_embedder():
    """Lazy singleton, same pattern as the LLM getters above -- avoids
    loading the (~1.5GB) embedding model for a query that never needs the
    prefilter (e.g. a scoped single-document query)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _prefilter_trees(question: str, tree_paths: List[Path], top_n: int = _TOP_N_CANDIDATES) -> List[Path]:
    """Local, free, no-LLM-call narrowing of which trees are even worth an
    LLM look -- cosine similarity between the question and each tree's own
    top-level section summaries (already written at ingestion), using the
    same embedding model already loaded for vector search. Verified live:
    firing an LLM call per tree (45+ trees in this corpus) was the actual
    bottleneck for an unscoped query (4.5+ minutes, ~130 calls); this step
    is a single local encode pass plus a numpy comparison, effectively
    free and near-instant."""
    if len(tree_paths) <= top_n:
        return tree_paths
    import numpy as np

    model = _get_embedder()
    question_embedding = model.encode([question])[0]

    scored = []
    for tp in tree_paths:
        tree = _load_tree(tp)
        summary_text = " ".join(s.get("summary") or "" for s in tree.get("sections", []))
        if not summary_text.strip():
            scored.append((tp, 0.0))
            continue
        tree_embedding = model.encode([summary_text])[0]
        similarity = float(
            np.dot(question_embedding, tree_embedding)
            / (np.linalg.norm(question_embedding) * np.linalg.norm(tree_embedding) + 1e-9)
        )
        scored.append((tp, similarity))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [tp for tp, _ in scored[:top_n]]


def _tree_path(path: Path) -> Path:
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)
    return TREES_DIR / f"{safe_stem}.json"


def _summarize(text: str, title: str) -> str:
    text = text.strip()
    if not text:
        return ""
    prompt = (
        f"Summarize the following document section in 1-2 sentences, capturing what a reader "
        f"would find here so they can decide if it's relevant to their question.\n\n"
        f"Section title: {title}\n\nSection text:\n{text[:3000]}\n\nSummary:"
    )
    try:
        return _get_pageindex_llm().invoke(prompt).content.strip()
    except Exception as e:
        print(f"[ERROR] Page-index summarization failed for section '{title}': {e}")
        return text[:200]


def _summarize_many(items: List[tuple[str, str]]) -> List[str]:
    """Concurrent version of _summarize for a batch of (text, title) pairs.
    Each call is an independent Groq round-trip -- building a tree with N
    sections used to mean N sequential waits; a small thread pool lets them
    overlap instead (pool.map preserves input order, so results line up
    with items positionally)."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
        return list(pool.map(lambda item: _summarize(*item), items))


def _sections_from_pdf_toc(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Use the PDF's own bookmark/outline structure when present -- far more
    reliable than text heuristics since it reflects the author's actual
    section boundaries, including exact page ranges."""
    doc = pymupdf.open(str(path))
    try:
        toc = doc.get_toc()  # [(level, title, page_1_indexed), ...]
        if not toc:
            return None
        total_pages = doc.page_count
        # Only keep top-level entries for the tree's first layer -- deeper
        # TOC levels would make the walk's branching factor unpredictable
        # across documents; MAX_WALK_DEPTH already bounds navigation depth
        # instead of relying on the source TOC's own nesting.
        top_level = min(lvl for lvl, _, _ in toc)
        entries = [(title, page) for lvl, title, page in toc if lvl == top_level]
        sections = []
        for i, (title, start_page) in enumerate(entries):
            end_page = entries[i + 1][1] - 1 if i + 1 < len(entries) else total_pages
            end_page = max(end_page, start_page)
            page_text = "\n".join(
                doc[p - 1].get_text() for p in range(start_page, min(end_page, total_pages) + 1)
            )
            sections.append(
                {"title": title, "page_start": start_page, "page_end": end_page, "text": page_text}
            )
        return sections
    finally:
        doc.close()


def _sections_from_heading_text(docs: List[Document]) -> List[Dict[str, Any]]:
    """Fallback when no PDF outline exists (or for Word docs): split on
    heading-like lines detected the same way classifier.py decides
    page-index routing in the first place, so a document that qualified for
    this route always has at least one section boundary to split on."""
    full_text = "\n".join(d.page_content for d in docs)
    lines = full_text.splitlines()

    sections: List[Dict[str, Any]] = []
    current_title = "Introduction"
    current_lines: List[str] = []

    def _flush():
        if current_lines:
            sections.append({"title": current_title, "page_start": -1, "page_end": -1, "text": "\n".join(current_lines)})

    for line in lines:
        stripped = line.strip()
        is_heading = stripped and not SENTENCE_END_RE.search(stripped) and HEADING_LINE_RE.match(stripped)
        if is_heading:
            _flush()
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)
    _flush()

    return sections if sections else [{"title": "Full document", "page_start": -1, "page_end": -1, "text": full_text}]


# A chapter/section larger than this gets a further sub-split before
# summarization -- verified live: a whole-chapter section (11K-17K chars)
# made narrow single-fact questions ("What are Stop Sequences?") fail the
# groundedness guardrail even when the generated answer was correct, because
# the judge couldn't confirm word-level support against such a large, mixed
# blob. Sized well above a single Q&A entry so short/uniform chapters are
# left alone.
MAX_SECTION_CHARS = 4000

# Many source documents (this one included) are structured as numbered
# Q&A entries ("Q11. What is Temperature in an LLM?") rather than
# conventional ALL-CAPS/Title-Case headings -- HEADING_LINE_RE never matches
# these (they end in "?", which SENTENCE_END_RE disqualifies). Tried first
# since, when present, it's a far more reliable split point than the
# generic heading heuristic for this shape of document.
_QA_MARKER_RE = re.compile(r"^Q\d{1,3}\.\s+.+$", re.MULTILINE)


def _split_large_section(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split one section into finer sub-sections if it's larger than
    MAX_SECTION_CHARS. Falls back through: numbered Q&A markers -> generic
    heading lines -> leave unsplit (some chapters are legitimately short and
    uniform, and forcing a split there would just fragment prose)."""
    text = section["text"]
    if len(text) <= MAX_SECTION_CHARS:
        return [section]

    def _sub(title: str, body: str) -> Dict[str, Any]:
        return {
            "title": title,
            # Sub-sections don't have their own page boundaries once split
            # from raw chapter text -- approximate with the parent
            # chapter's range rather than inventing precision we don't have.
            "page_start": section["page_start"],
            "page_end": section["page_end"],
            "text": body,
        }

    qa_matches = list(_QA_MARKER_RE.finditer(text))
    if len(qa_matches) >= 2:
        subs = []
        for i, m in enumerate(qa_matches):
            start = m.start()
            end = qa_matches[i + 1].start() if i + 1 < len(qa_matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            heading_line = m.group().strip()
            subs.append(_sub(f"{section['title']} — {heading_line}", body))
        if subs:
            return subs

    lines = text.splitlines()
    subs = []
    current_title = section["title"]
    current_lines: List[str] = []

    def _flush():
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                subs.append(_sub(current_title, body))

    for line in lines:
        stripped = line.strip()
        is_heading = stripped and not SENTENCE_END_RE.search(stripped) and HEADING_LINE_RE.match(stripped)
        if is_heading:
            _flush()
            current_title = f"{section['title']} — {stripped}"
            current_lines = []
        else:
            current_lines.append(line)
    _flush()

    return subs if len(subs) >= 2 else [section]


def build_tree(path: Path, docs: List[Document]) -> str:
    """Build and persist a section tree for one document. Returns the tree
    file path (as a string, for manifest tracking in sync.py)."""
    sections = None
    if path.suffix.lower() == ".pdf":
        sections = _sections_from_pdf_toc(path)
    if sections is None:
        sections = _sections_from_heading_text(docs)

    # Nest sub-sections under their parent chapter instead of flattening
    # everything into one list -- verified live: a flat list of 111
    # sub-sections made _pick_section's single "here are all the summaries,
    # pick one" prompt too large for Groq's per-minute input-token limit
    # (ITPM 7000, needed ~9700), when the tree used to have only 17 entries.
    # Nesting keeps each individual pick prompt small (~17 chapters, then
    # ~5-10 children of whichever chapter matched) by reusing the two-level
    # walk that query()/_pick_section already support via MAX_WALK_DEPTH.
    nodes = []
    # Every _summarize() call this tree needs -- one per unsplit section,
    # one per child of a split chapter -- gets queued here instead of called
    # inline, so _summarize_many() below can run them all concurrently in
    # one batch rather than one-at-a-time as the loop reaches each section.
    summary_queue: List[tuple[str, str, dict]] = []

    for s in sections:
        if not s["text"].strip():
            continue
        subs = _split_large_section(s)
        if len(subs) == 1:
            node = {
                "title": s["title"],
                "page_start": s["page_start"],
                "page_end": s["page_end"],
                "summary": None,  # filled in after the concurrent batch below
                "text": s["text"],
            }
            nodes.append(node)
            summary_queue.append((s["text"], s["title"], node))
        else:
            children = [
                {
                    "title": sub["title"],
                    "page_start": sub["page_start"],
                    "page_end": sub["page_end"],
                    "summary": None,
                    "text": sub["text"],
                }
                for sub in subs
                if sub["text"].strip()
            ]
            live_subs = [sub for sub in subs if sub["text"].strip()]
            for sub, child in zip(live_subs, children):
                summary_queue.append((sub["text"], sub["title"], child))
            # The chapter's own summary must list every child topic, not
            # describe the chapter's raw text -- verified live: _summarize()
            # only sees the first 3000 characters of the source text, so for
            # an 11K-17K character chapter it silently dropped any topic
            # appearing later (e.g. "pre-training" and "fine-tuning" showed
            # up after char 3000 in Chapter 1), causing the top-level
            # navigator to skip or misroute past that chapter entirely since
            # its summary never mentioned them. Building the summary from
            # the child list instead is free (no extra LLM call) and exact,
            # since it's just the topics we already split out.
            topics = "; ".join(child["title"].split(" — ")[-1] for child in children)
            nodes.append(
                {
                    "title": s["title"],
                    "page_start": s["page_start"],
                    "page_end": s["page_end"],
                    "summary": f"Covers: {topics}",
                    "text": s["text"],
                    "children": children,
                }
            )

    summaries = _summarize_many([(text, title) for text, title, _ in summary_queue])
    for (_, _, target), summary in zip(summary_queue, summaries):
        target["summary"] = summary

    tree = {"source": str(path), "source_file": path.name, "sections": nodes}
    tree_path = _tree_path(path)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(json.dumps(tree, indent=2))
    print(f"[PAGEINDEX] Built tree for {path.name}: {len(nodes)} sections -> {tree_path}")
    return str(tree_path)


def delete_tree(tree_path: str) -> None:
    """Used by sync.py when a vectorless_pageindex-routed file is modified
    (delete-then-rebuild) or removed."""
    p = Path(tree_path)
    if p.exists():
        p.unlink()


def list_trees() -> List[Path]:
    if not TREES_DIR.exists():
        return []
    return sorted(TREES_DIR.glob("*.json"))


def _load_tree(tree_path: Path) -> Dict[str, Any]:
    return json.loads(tree_path.read_text())


def _pick_section(question: str, sections: List[Dict[str, Any]]) -> Optional[int]:
    """One LLM call: given section summaries, pick the single most relevant
    one (or none) for this question. Returns an index into sections, or
    None if nothing looks relevant."""
    if not sections:
        return None
    # The index marker uses [brackets] rather than a bare "N." -- verified
    # live: a section title carrying the source document's own numbering
    # (e.g. "...Q07. What is pre-training?") sat right next to a bare list
    # index on the same line ("6. ...Q07...") and the model picked the
    # title's number instead of the list index, landing one entry off. The
    # prompt now explicitly calls out that distinction too, not just the
    # format change, since a title could carry other numbers as well.
    listing = "\n".join(f"[{i}] {s['title']}: {s['summary']}" for i, s in enumerate(sections))
    prompt = (
        "Given the question and this list of document sections, respond with ONLY the bracketed "
        "index number of the single most relevant section (e.g. `3`), or `none` if none of them "
        "are relevant. Each section's index is the number in [brackets] at the start of its line -- "
        "ignore any other numbers that appear inside a section's own title or summary (such as a "
        "question number like \"Q07\"), those are unrelated to its index.\n\n"
        f"Sections:\n{listing}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        answer = _get_pageindex_picker_llm().invoke(prompt).content.strip().lower()
    except Exception as e:
        print(f"[ERROR] Page-index section selection failed: {e}")
        return None
    match = re.search(r"\d+", answer)
    if not match:
        return None
    idx = int(match.group())
    return idx if 0 <= idx < len(sections) else None


def _query_tree(question: str, tree_path: Path) -> Optional[Dict[str, Any]]:
    """Navigate one document's tree top-down (bounded by MAX_WALK_DEPTH) and
    return its matched section as a retrieved-document dict, or None if
    nothing in this tree matched."""
    tree = _load_tree(tree_path)
    remaining = tree.get("sections", [])
    # Descend level by level: pick among the current list, then if the
    # picked node has children (a chapter that was split into finer
    # sub-sections), pick again among just those children instead of
    # returning the whole chapter blob. Keeps each individual prompt
    # small -- e.g. ~17 chapters, then only that chapter's ~5-10
    # children -- rather than listing every leaf section in one go
    # (verified live: doing that for 111 flat leaves exceeded Groq's
    # per-minute input-token limit).
    section = None
    for _ in range(MAX_WALK_DEPTH):
        idx = _pick_section(question, remaining)
        if idx is None:
            section = None
            break
        section = remaining[idx]
        children = section.get("children")
        if not children:
            break
        remaining = children
    if section is None:
        return None
    return {
        "content": section["text"],
        "metadata": {
            "source_file": tree.get("source_file", tree_path.name),
            "file_type": "pageindex_section",
            "page": section.get("page_start", -1),
            "section_title": section["title"],
        },
        "score": None,
    }


def query(question: str, tree_paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Navigate each document's tree top-down and return the matched
    section's full text as retrieved content, shaped like the vector path's
    output so graph.py's generate() node can consume either uniformly.
    Each tree's walk is fully independent of every other tree's, so an
    unscoped ("search everything") query -- which fans out across every
    page-index tree -- runs them concurrently rather than one document at a
    time."""
    candidates = tree_paths if tree_paths is not None else list_trees()
    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
        results = pool.map(lambda tp: _query_tree(question, tp), candidates)
    return [r for r in results if r is not None]


__all__ = ["ROUTING_PAGEINDEX", "build_tree", "delete_tree", "list_trees", "query"]
