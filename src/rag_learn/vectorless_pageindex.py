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

# Hard cap on how many individual sections (not documents) ever reach the
# LLM relevance call -- ranked globally across every candidate document's
# top-level sections, not per-document. Verified live: ranking at the
# document level (a fixed number of documents, every one of that document's
# sections included) still let a single Q&A-style document with hundreds of
# top-level sections blow past Groq's 7,000-input-tokens-per-minute limit
# for this model, even after chunking the LLM call into smaller requests --
# chunking only avoids one oversized request, it does not fix the account's
# total per-minute throughput ceiling shared across all those chunks. 25
# short section listings (title + summary) comfortably fits in one call
# under that budget.
_TOP_N_SECTIONS = 25

# Coarse first pass: how many documents survive before the expensive
# section-level embedding step even runs. Verified live: this corpus's 45
# documents contain 4,378 sections total (Q&A-style documents split into
# hundreds each) -- embedding all 4,378 against the question took 144.5s on
# this hardware, blowing the time budget even with zero LLM calls involved.
# One cheap embedding per document (not per section) first narrows the
# field before paying the per-section cost only within the survivors.
_TOP_N_TREES = 15

# Backstop only -- with _TOP_N_SECTIONS this small, the flattened listing
# should always fit in a single chunk, but this keeps any single call from
# exceeding Groq's per-minute input-token limit if that assumption ever
# breaks (e.g. _TOP_N_SECTIONS raised later, or unusually long summaries).
_BATCH_CHUNK_SIZE = 40


def _get_embedder():
    """Lazy singleton, same pattern as the LLM getters above -- avoids
    loading the (~1.5GB) embedding model for a query that never needs the
    prefilter (e.g. a scoped single-document query)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _cosine_similarities(question_embedding, candidate_embeddings) -> "Any":
    import numpy as np

    norms = np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-9
    return (candidate_embeddings @ question_embedding) / norms


def _prefilter_sections(
    question: str,
    tree_paths: List[Path],
    top_n: int = _TOP_N_SECTIONS,
    top_n_trees: int = _TOP_N_TREES,
) -> tuple[Dict[Path, Dict[str, Any]], List[Dict[str, Any]], List[tuple[Path, int]]]:
    """Local, free, no-LLM-call ranking, in two passes so the expensive pass
    only runs on a shortlist:

    1. Coarse: one embedding per document (its own top-level summaries,
       concatenated), batch-encoded together, kept to the top_n_trees most
       similar documents. Verified live: this corpus's 4,378 total
       sections took 144.5s to embed individually -- comparing whole
       documents first (45 embeddings, not 4,378) is cheap enough to run
       unconditionally and prunes the field before the expensive part.
    2. Fine: within that shortlist only, rank each individual top-level
       section directly and keep the top_n most similar OVERALL regardless
       of which surviving document it came from -- so a single Q&A-style
       document with many sections still can't flood the LLM listing with
       only its own sections (see _TOP_N_SECTIONS's docstring).

    Returns (loaded trees by path -- shortlisted ones only, the top_n
    section dicts, their (tree_path, local_index) origins) so the caller
    can route an LLM's answer back to the right document without reloading
    anything."""
    all_trees = {tp: _load_tree(tp) for tp in tree_paths}

    model = _get_embedder()
    question_embedding = model.encode([question])[0]

    if len(all_trees) > top_n_trees:
        tree_items = list(all_trees.items())
        # Truncated -- verified live: a Q&A-style document with hundreds of
        # sections, concatenated untruncated, produced a sequence long
        # enough that the embedding model's attention-mask allocation
        # crashed outright (RuntimeError: invalid buffer size, 17.56 GiB).
        # 2000 chars is plenty of signal for "is this document even in the
        # right topic area" -- the fine section-level pass below is what
        # actually judges individual sections precisely.
        tree_texts = [
            (" ".join(s.get("summary") or "" for s in t.get("sections", [])) or t.get("source_file", ""))[:2000]
            for _, t in tree_items
        ]
        tree_embeddings = model.encode(tree_texts, batch_size=32)
        tree_similarities = _cosine_similarities(question_embedding, tree_embeddings)
        ranked_tree_idx = sorted(range(len(tree_items)), key=lambda i: tree_similarities[i], reverse=True)[
            :top_n_trees
        ]
        trees = {tree_items[i][0]: tree_items[i][1] for i in ranked_tree_idx}
    else:
        trees = all_trees

    flat_sections: List[Dict[str, Any]] = []
    origin: List[tuple[Path, int]] = []
    for tp, tree in trees.items():
        for i, s in enumerate(tree.get("sections", [])):
            flat_sections.append(s)
            origin.append((tp, i))

    if len(flat_sections) <= top_n:
        return trees, flat_sections, origin

    # Same safeguard as the coarse pass above (2000-char cap): an
    # unsummarized or runaway-length summary here, multiplied across a
    # 1,000+-section shortlist and batched by encode(), is what produced a
    # verified 462GB allocation and hang -- cap text length and force a
    # bounded batch_size so no single encode() call sees an unbounded batch.
    texts = [f"{s['title']}: {s.get('summary') or ''}"[:2000] for s in flat_sections]
    section_embeddings = model.encode(texts, batch_size=32)
    similarities = _cosine_similarities(question_embedding, section_embeddings)

    ranked = sorted(range(len(flat_sections)), key=lambda i: similarities[i], reverse=True)[:top_n]
    return trees, [flat_sections[i] for i in ranked], [origin[i] for i in ranked]


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


_pageindex_batch_llm = None


def _get_pageindex_batch_llm():
    global _pageindex_batch_llm
    if _pageindex_batch_llm is None:
        # No fallback chain (get_llm_groq_only) -- see _pick_sections_batch's
        # docstring for why a slow free-model fallback isn't worth it here.
        _pageindex_batch_llm = config.get_llm_groq_only(
            temperature=0.0, max_tokens=60, purpose="pageindex_batch_pick"
        )
    return _pageindex_batch_llm


def _pick_sections_batch(question: str, sections: List[Dict[str, Any]]) -> List[int]:
    """Batched version of _pick_section: one LLM call judging ALL candidate
    sections at once (mirrors graph.grade_documents' batching), returning
    EVERY plausibly relevant index rather than a single best match -- a
    broad/definitional question (e.g. "What is RAG?") legitimately has a
    paragraph-level answer spread across several different documents rather
    than being any one document's dedicated chapter, and keeping only one
    match systematically under-serves that case.

    On failure, skips this batch (returns no matches from it) rather than
    keeping every candidate -- verified live: with a large/uneven corpus
    (some documents split into hundreds of top-level Q&A-style sections
    rather than a handful of chapters), "keep everything on failure"
    degraded to returning ~1400 unfiltered results, defeating the entire
    point of filtering. query() calls this per token-budget-sized chunk
    (see _BATCH_CHUNK_SIZE), so skipping one failed chunk only drops that
    slice of candidates, not the whole query's results."""
    if not sections:
        return []
    listing = "\n".join(f"[{i}] {s['title']}: {s['summary']}" for i, s in enumerate(sections))
    prompt = (
        "Given the question and this list of document sections, respond with ONLY a "
        "comma-separated list of the bracketed index numbers of every section that looks "
        "at least plausibly relevant (e.g. `0,2,3`), or `none` if none are. Err toward "
        "including a section if it's a reasonable candidate -- a downstream step will "
        "double check. Each section's index is the number in [brackets] at the start of "
        "its line -- ignore any other numbers inside a section's own title or summary.\n\n"
        f"Sections:\n{listing}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        answer = _get_pageindex_batch_llm().invoke(prompt).content.strip().lower()
    except Exception as e:
        print(f"[ERROR] Page-index batch section selection failed, skipping this chunk: {e}")
        return []
    if answer == "none":
        return []
    indices: set[int] = set()
    for tok in answer.replace(" ", "").split(","):
        digits = re.sub(r"[^\d]", "", tok)
        if digits and 0 <= int(digits) < len(sections):
            indices.add(int(digits))
    return sorted(indices)


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
    """Navigate page-index trees and return matched sections' full text,
    shaped like the vector path's output so graph.py's generate() node can
    consume either uniformly.

    Scoped call (tree_paths given, from graph._match_pageindex_tree's
    "search this document" mode): unchanged behavior, one tree, the
    existing single-pick walk (_query_tree).

    Unscoped call ("search everything", tree_paths=None): this is the path
    that was observed live to fire 45+ individual LLM calls and take 4.5+
    minutes. A free local-embedding prefilter ranks every candidate
    document's individual sections directly and keeps only the top
    _TOP_N_SECTIONS overall (_prefilter_sections -- NOT a fixed number of
    documents, since a handful of Q&A-style documents in this corpus split
    into hundreds of top-level sections each, which blew past Groq's
    per-minute token budget when document-level shortlisting was tried).
    ONE batched LLM call across that shortlist (_pick_sections_batch)
    returns every plausibly relevant match, not just one -- letting several
    different documents' perspectives on the same topic survive into
    generate()."""
    candidates = tree_paths if tree_paths is not None else list_trees()
    if not candidates:
        return []

    if tree_paths is not None:
        with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
            results = pool.map(lambda tp: _query_tree(question, tp), candidates)
        return [r for r in results if r is not None]

    trees, flat_sections, origin = _prefilter_sections(question, candidates)

    # Chunk the (already small, _TOP_N_SECTIONS-capped) listing so no single
    # call exceeds Groq's per-minute input-token limit -- a backstop, not
    # the primary defense (see _BATCH_CHUNK_SIZE's comment).
    chunk_starts = list(range(0, len(flat_sections), _BATCH_CHUNK_SIZE))
    with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
        chunk_results = pool.map(
            lambda start: _pick_sections_batch(question, flat_sections[start : start + _BATCH_CHUNK_SIZE]),
            chunk_starts,
        )
    matched_indices = [
        start + local_idx for start, local_indices in zip(chunk_starts, chunk_results) for local_idx in local_indices
    ]

    results: List[Dict[str, Any]] = []
    for idx in matched_indices:
        tp, _ = origin[idx]
        section = flat_sections[idx]
        tree = trees[tp]
        children = section.get("children")
        if children:
            # Bounded, low-volume (at most _TOP_N_SECTIONS of these) --
            # the existing single-pick walk is fine at this depth.
            child_idx = _pick_section(question, children)
            section = children[child_idx] if child_idx is not None else section
        results.append(
            {
                "content": section["text"],
                "metadata": {
                    "source_file": tree.get("source_file", tp.name),
                    "file_type": "pageindex_section",
                    "page": section.get("page_start", -1),
                    "section_title": section["title"],
                },
                "score": None,
            }
        )
    return results


__all__ = ["ROUTING_PAGEINDEX", "build_tree", "delete_tree", "list_trees", "query"]
