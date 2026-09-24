"""Document classification: tags every ingested source file with a
`routing` value so later phases can send it down the right retrieval path
instead of forcing everything through embeddings/vector search.

See Phase 2.5 in the implementation plan for the full design/rationale.
Three routes:
- "vector"              -- flat/narrative content, classic embedding search.
- "vectorless_sql"      -- tabular data, queried directly (Phase 3b.i).
- "vectorless_pageindex" -- long structured documents, tree-navigated (Phase 3b.ii).

This module only classifies and tags -- the actual SQL/page-index query
paths are Phase 3b's job, not this one.
"""

import re
from pathlib import Path
from typing import List, Optional

import pymupdf
from langchain_core.documents import Document

from rag_learn.llm_factory import default_factory

ROUTING_VECTOR = "vector"
ROUTING_SQL = "vectorless_sql"
ROUTING_PAGEINDEX = "vectorless_pageindex"

# File type alone is sufficient signal for tabular data -- no ambiguity, so
# no structure check needed (matches the plan's "Tabular files ... always
# vectorless_sql").
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# PDFs and Word docs get a structure check (headings/TOC) since either a
# flat résumé-style document or a heavily-sectioned guide can share this
# extension. Text/JSON are flat/narrative by nature -- no check needed.
STRUCTURE_CHECKED_EXTENSIONS = {".pdf", ".docx"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Below this many heading-like lines, a document reads as flat/narrative
# rather than sectioned -- chosen as "more than a coincidental capitalized
# line or two, fewer than what a real single-topic résumé would ever have."
MIN_HEADING_LINES_FOR_PAGEINDEX = 3

# A heading-like line: short (title/section headers aren't paragraphs),
# doesn't end in sentence punctuation, and is either ALL CAPS or Title Case
# -- catches common heading styles across plain-text extraction from PDFs
# and Word docs, where original font-size/style info is lost.
HEADING_LINE_RE = re.compile(
    r"^(?:[A-Z][A-Za-z0-9 ,'&/-]{2,60}|[A-Z0-9 ,'&/-]{3,60})$"
)
SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

# Image-text heuristic for the hybrid classifier (Phase 2.5 rationale: catch
# obvious cases for free, only pay for an LLM call on genuinely ambiguous
# text). A grid/table reads as many lines with a similar number of
# whitespace-or-delimiter-separated tokens; flowing prose reads as long
# lines with widely varying token counts and few explicit delimiters.
_TABLE_DELIMITER_RE = re.compile(r"[\t|]|(?: {2,})")
_MIN_LINES_FOR_TABLE_HEURISTIC = 3


def _has_pdf_toc(path: Path) -> bool:
    try:
        doc = pymupdf.open(str(path))
        try:
            return len(doc.get_toc()) > 0
        finally:
            doc.close()
    except Exception as e:
        print(f"[ERROR] Failed to read PDF outline for {path.name}: {e}")
        return False


def _looks_structured_by_headings(text: str) -> bool:
    heading_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or SENTENCE_END_RE.search(stripped):
            continue
        if HEADING_LINE_RE.match(stripped):
            heading_count += 1
    return heading_count >= MIN_HEADING_LINES_FOR_PAGEINDEX


def _looks_tabular_heuristic(text: str) -> Optional[bool]:
    """Returns True (tabular), False (narrative), or None (ambiguous --
    caller should fall back to an LLM call)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < _MIN_LINES_FOR_TABLE_HEURISTIC:
        return None  # too little text to judge either way

    delimited_lines = sum(1 for line in lines if _TABLE_DELIMITER_RE.search(line))
    token_counts = [len(line.split()) for line in lines]
    avg_tokens = sum(token_counts) / len(token_counts)

    # Most lines carry explicit column delimiters (tabs, pipes, aligned
    # multi-space gaps) -- a strong, cheap tabular signal.
    if delimited_lines / len(lines) >= 0.6:
        return True

    # Long flowing lines with no delimiters at all -- confidently narrative
    # prose, not worth an LLM call to confirm.
    if delimited_lines == 0 and avg_tokens >= 8:
        return False

    return None  # short lines, no delimiters, or a mixed signal -- ambiguous


def _classify_image_with_llm(text: str) -> str:
    prompt = (
        "The following text was OCR'd from an image. Classify it as exactly one "
        "word: `table` if it is primarily tabular/structured data (rows and "
        "columns, forms, spreadsheets), or `narrative` if it is primarily prose, "
        "a document page, a chat/social post, or a diagram's descriptive text.\n\n"
        f"Text:\n{text[:2000]}\n\nClassification:"
    )
    try:
        # max_tokens: a one-word classification call can otherwise request far
        # more output headroom than needed and get rejected by Groq's quota.
        llm = default_factory.get("classification", temperature=0.0, max_tokens=20)
        answer = llm.invoke(prompt).content.strip().lower()
    except Exception as e:
        print(f"[ERROR] Image classification LLM call failed, defaulting to vector: {e}")
        return ROUTING_VECTOR
    return ROUTING_SQL if "table" in answer else ROUTING_VECTOR


def classify_document(path: Path, docs: List[Document]) -> str:
    """Decide the routing for one source file, given its already-loaded
    Document objects (used for text-based heuristics -- avoids re-reading
    the file from disk)."""
    ext = path.suffix.lower()

    if ext in TABULAR_EXTENSIONS:
        return ROUTING_SQL

    if ext in IMAGE_EXTENSIONS:
        text = "\n".join(d.page_content for d in docs)
        heuristic = _looks_tabular_heuristic(text)
        if heuristic is True:
            return ROUTING_SQL
        if heuristic is False:
            return ROUTING_VECTOR
        return _classify_image_with_llm(text)

    if ext in STRUCTURE_CHECKED_EXTENSIONS:
        if ext == ".pdf" and _has_pdf_toc(path):
            return ROUTING_PAGEINDEX
        text = "\n".join(d.page_content for d in docs)
        if _looks_structured_by_headings(text):
            return ROUTING_PAGEINDEX
        return ROUTING_VECTOR

    # .txt, .json, and any other flat/narrative-by-nature type.
    return ROUTING_VECTOR


def classify_and_tag(path: Path, docs: List[Document]) -> str:
    """Classify path's documents and stamp the routing decision onto every
    Document's metadata. Returns the routing value for logging/counting."""
    if not docs:
        return ROUTING_VECTOR
    routing = classify_document(path, docs)
    for doc in docs:
        doc.metadata["routing"] = routing
    print(f"[DEBUG] Routing: {path.name} -> {routing}")
    return routing
