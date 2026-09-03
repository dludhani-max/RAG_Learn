from pathlib import Path
from typing import Any, List

import pdfplumber
import pymupdf
import pytesseract
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document
from PIL import Image

from rag_learn import config

# Below this many native-extracted characters, a PDF page is treated as
# scanned/image-only and OCR'd instead (see Q&A AI PM.pdf: 171 pages,
# ~3 chars/page natively, real content recovered via page-image OCR).
PDF_OCR_CHAR_THRESHOLD = 50

# Bump when loader logic materially changes (e.g. the per-page OCR fallback
# and table extraction added here) -- sync.py folds this into its pipeline
# fingerprint so existing files get reprocessed under the new logic instead
# of being skipped as "unchanged" (their content_hash is unaffected by a
# loader code change, so hash alone wouldn't catch this).
LOADER_VERSION = 2


def _tag(doc: Document, path: Path, file_type: str, **extra: Any) -> Document:
    doc.metadata["source_file"] = path.name
    doc.metadata["source"] = str(path)
    doc.metadata["file_type"] = file_type
    doc.metadata.update(extra)
    return doc


MAX_TABLE_CELL_CHARS = 150


def _is_real_table(table: List[List[Any]]) -> bool:
    """pdfplumber's heuristic detector produces false positives on aligned
    code/paragraph text and multi-column page layouts (e.g. a 1x2 'table'
    from indentation, or two text columns misread as a 2-col table). Require
    a genuine grid: >=2 rows, >=2 columns, mostly non-empty cells, and cells
    short enough to be table values rather than flowing paragraph text
    (verified: real tables in this corpus max out at ~83 chars/cell, while
    false-positive 'tables' from page layout run 200-800+ chars/cell)."""
    if len(table) < 2 or len(table[0]) < 2:
        return False
    total = sum(len(row) for row in table)
    non_empty = sum(1 for row in table for cell in row if cell and str(cell).strip())
    if total == 0 or (non_empty / total) < 0.5:
        return False
    max_cell_len = max((len(str(cell)) for row in table for cell in row if cell), default=0)
    return max_cell_len <= MAX_TABLE_CELL_CHARS


def _table_to_markdown(table: List[List[Any]]) -> str:
    rows = [[("" if cell is None else str(cell)).replace("\n", " ") for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _load_pdf(path: Path) -> List[Document]:
    """Per-page extraction: native text where available, OCR fallback for
    scanned/image-only pages, plus local (pdfplumber, free/no-LLM) table
    detection appended as markdown alongside the page's prose text."""
    docs: List[Document] = []
    fitz_doc = pymupdf.open(str(path))
    try:
        with pdfplumber.open(str(path)) as plumber_doc:
            for i, page in enumerate(fitz_doc):
                native_text = page.get_text()
                if len(native_text.strip()) < PDF_OCR_CHAR_THRESHOLD:
                    pix = page.get_pixmap(dpi=200)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    content = pytesseract.image_to_string(image)
                    extraction_method = "ocr"
                else:
                    content = native_text
                    extraction_method = "native"

                has_table = False
                if i < len(plumber_doc.pages):
                    try:
                        tables = plumber_doc.pages[i].extract_tables()
                    except Exception as e:
                        tables = []
                        print(f"[ERROR] Table extraction failed on {path.name} page {i}: {e}")
                    md_tables = [_table_to_markdown(t) for t in tables if _is_real_table(t)]
                    if md_tables:
                        has_table = True
                        content = content + "\n\n" + "\n\n".join(md_tables)

                doc = Document(page_content=content)
                docs.append(
                    _tag(doc, path, "pdf", page=i, extraction_method=extraction_method, has_table=has_table)
                )
    finally:
        fitz_doc.close()
    return docs


def _load_text(path: Path) -> List[Document]:
    return [_tag(d, path, "text") for d in TextLoader(str(path)).load()]


def _load_csv(path: Path) -> List[Document]:
    return [_tag(d, path, "csv") for d in CSVLoader(str(path)).load()]


def _load_excel(path: Path) -> List[Document]:
    return [_tag(d, path, "excel") for d in UnstructuredExcelLoader(str(path), mode="elements").load()]


def _load_word(path: Path) -> List[Document]:
    return [_tag(d, path, "word") for d in Docx2txtLoader(str(path)).load()]


def _load_json(path: Path) -> List[Document]:
    return [_tag(d, path, "json") for d in JSONLoader(str(path), jq_schema=".", text_content=False).load()]


def _load_image(path: Path) -> List[Document]:
    text = pytesseract.image_to_string(Image.open(path))
    if not text.strip():
        return []
    return [_tag(Document(page_content=text), path, "image")]


# Extension -> (file_type label, per-file loader). Shared by load_all_documents
# (directory glob) and sync.py (single-file reprocessing on add/modify).
LOADERS = {
    ".pdf": ("pdf", _load_pdf),
    ".txt": ("text", _load_text),
    ".csv": ("csv", _load_csv),
    ".xlsx": ("excel", _load_excel),
    ".xls": ("excel", _load_excel),
    ".docx": ("word", _load_word),
    ".json": ("json", _load_json),
    ".jpg": ("image", _load_image),
    ".jpeg": ("image", _load_image),
    ".png": ("image", _load_image),
}


def load_document(path: Path) -> List[Document]:
    """Load a single file, dispatching by extension. Used both by
    load_all_documents (bulk) and sync.py (per-file re-ingestion)."""
    entry = LOADERS.get(path.suffix.lower())
    if entry is None:
        return []
    _, loader_fn = entry
    try:
        return loader_fn(path)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return []


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load supported files from the data directory and convert to LangChain
    Document objects. Supported: PDF (with per-page OCR fallback and table
    detection), TXT, CSV, Excel, Word, JSON, images (OCR).

    Each file is loaded independently so one bad file doesn't abort the
    whole ingestion run.
    """
    data_path = Path(data_dir).resolve()
    vector_store_dir = Path(config.VECTOR_STORE_DIR).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents: List[Any] = []
    counts: dict[str, int] = {}
    for path in sorted(data_path.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        if vector_store_dir in path.resolve().parents:
            continue  # never ingest the vector store's own state files
        file_type, _ = LOADERS[path.suffix.lower()]
        loaded = load_document(path)
        counts[file_type] = counts.get(file_type, 0) + len(loaded)
        documents.extend(loaded)
        print(f"[DEBUG] Loaded {len(loaded)} docs from {file_type}: {path.name}")

    print(f"[DEBUG] Ingestion summary: {counts} | total documents: {len(documents)}")
    return documents
