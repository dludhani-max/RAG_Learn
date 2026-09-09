import hashlib
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
from rag_learn.classifier import classify_and_tag

# Below this many native-extracted characters, a PDF page is treated as
# scanned/image-only and OCR'd instead (see Q&A AI PM.pdf: 171 pages,
# ~3 chars/page natively, real content recovered via page-image OCR).
PDF_OCR_CHAR_THRESHOLD = 50

# Embedded images smaller than this (either dimension) are treated as
# icons/bullets/decorative elements, not genuine diagrams worth surfacing
# back to the user (verified against this corpus: real diagrams run
# 1000px+, decorative elements in QA AI.pdf were 200x62 / 500x300).
MIN_DIAGRAM_DIMENSION = 150

EXTRACTED_IMAGES_DIR = Path(config.VECTOR_STORE_DIR) / "extracted_images"

# Bump when loader logic materially changes (e.g. the per-page OCR fallback,
# table extraction, and embedded-diagram extraction added here) -- sync.py
# folds this into its pipeline fingerprint so existing files get
# reprocessed under the new logic instead of being skipped as "unchanged"
# (their content_hash is unaffected by a loader code change, so hash alone
# wouldn't catch this).
LOADER_VERSION = 3


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


def _extract_diagrams(
    fitz_doc, page, path: Path, page_index: int, seen_hashes: set
) -> List[str]:
    """Save each embedded raster image on this page to disk, skipping tiny
    icons/decorative elements. Returns saved file paths. Only called for
    natively-extracted pages -- an OCR-fallback page's "embedded image" is
    just the whole scanned page itself (already covered by the OCR render),
    not a distinct diagram.

    seen_hashes is shared across all pages of one document (populated by the
    caller): a cover banner or logo embedded verbatim on multiple pages
    passes the size filter (verified: QA AI.pdf's repeated 500x300 title
    banner appeared on 3 separate pages, indistinguishable by size alone
    from a real diagram) but is identical bytes every time, so content-hash
    dedup within the document catches it while a genuine diagram -- unique
    per page -- is unaffected."""
    saved: List[str] = []
    EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(page.get_images(full=True)):
        try:
            base = fitz_doc.extract_image(img[0])
        except Exception as e:
            print(f"[ERROR] Failed to extract image on {path.name} page {page_index}: {e}")
            continue
        if base.get("width", 0) < MIN_DIAGRAM_DIMENSION or base.get("height", 0) < MIN_DIAGRAM_DIMENSION:
            continue
        image_hash = hashlib.sha256(base["image"]).hexdigest()
        if image_hash in seen_hashes:
            continue
        seen_hashes.add(image_hash)
        out_path = EXTRACTED_IMAGES_DIR / f"{path.stem}__p{page_index}__{idx}.{base['ext']}"
        out_path.write_bytes(base["image"])
        saved.append(str(out_path))
    return saved


def _load_pdf(path: Path) -> List[Document]:
    """Per-page extraction: native text where available, OCR fallback for
    scanned/image-only pages, plus local (pdfplumber, free/no-LLM) table
    detection appended as markdown alongside the page's prose text."""
    docs: List[Document] = []
    fitz_doc = pymupdf.open(str(path))
    seen_image_hashes: set = set()
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

                images = (
                    _extract_diagrams(fitz_doc, page, path, i, seen_image_hashes)
                    if extraction_method == "native"
                    else []
                )

                doc = Document(page_content=content)
                docs.append(
                    _tag(
                        doc, path, "pdf", page=i, extraction_method=extraction_method,
                        has_table=has_table, images=images,
                    )
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
    # The file itself is the "diagram" here -- reference its own path so it
    # surfaces the same way an embedded PDF diagram would.
    return [_tag(Document(page_content=text), path, "image", images=[str(path)])]


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
    routing_counts: dict[str, int] = {}
    for path in sorted(data_path.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        if vector_store_dir in path.resolve().parents:
            continue  # never ingest the vector store's own state files
        file_type, _ = LOADERS[path.suffix.lower()]
        loaded = load_document(path)
        counts[file_type] = counts.get(file_type, 0) + len(loaded)
        if loaded:
            routing = classify_and_tag(path, loaded)
            routing_counts[routing] = routing_counts.get(routing, 0) + 1
        documents.extend(loaded)
        print(f"[DEBUG] Loaded {len(loaded)} docs from {file_type}: {path.name}")

    print(f"[DEBUG] Ingestion summary: {counts} | total documents: {len(documents)}")
    print(f"[DEBUG] Routing summary: {routing_counts}")
    return documents
