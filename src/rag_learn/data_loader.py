from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document
from PIL import Image
import pytesseract


def _tag(doc: Document, path: Path, file_type: str) -> Document:
    doc.metadata["source_file"] = path.name
    doc.metadata["source"] = str(path)
    doc.metadata["file_type"] = file_type
    return doc


def _load_pdfs(data_path: Path) -> List[Document]:
    docs: List[Document] = []
    for path in data_path.glob("**/*.pdf"):
        try:
            loaded = PyPDFLoader(str(path)).load()
            docs.extend(_tag(d, path, "pdf") for d in loaded)
            print(f"[DEBUG] Loaded {len(loaded)} pages from PDF: {path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {path}: {e}")
    return docs


def _load_text_files(data_path: Path) -> List[Document]:
    docs: List[Document] = []
    for path in data_path.glob("**/*.txt"):
        try:
            loaded = TextLoader(str(path)).load()
            docs.extend(_tag(d, path, "text") for d in loaded)
            print(f"[DEBUG] Loaded text file: {path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load text file {path}: {e}")
    return docs


def _load_csvs(data_path: Path) -> List[Document]:
    docs: List[Document] = []
    for path in data_path.glob("**/*.csv"):
        try:
            loaded = CSVLoader(str(path)).load()
            docs.extend(_tag(d, path, "csv") for d in loaded)
            print(f"[DEBUG] Loaded {len(loaded)} rows from CSV: {path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {path}: {e}")
    return docs


def _load_excel(data_path: Path) -> List[Document]:
    docs: List[Document] = []
    for pattern in ("**/*.xlsx", "**/*.xls"):
        for path in data_path.glob(pattern):
            try:
                loaded = UnstructuredExcelLoader(str(path), mode="elements").load()
                docs.extend(_tag(d, path, "excel") for d in loaded)
                print(f"[DEBUG] Loaded Excel file: {path.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load Excel file {path}: {e}")
    return docs


def _load_word(data_path: Path) -> List[Document]:
    docs: List[Document] = []
    for path in data_path.glob("**/*.docx"):
        try:
            loaded = Docx2txtLoader(str(path)).load()
            docs.extend(_tag(d, path, "word") for d in loaded)
            print(f"[DEBUG] Loaded Word file: {path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load Word file {path}: {e}")
    return docs


def _load_json(data_path: Path) -> List[Document]:
    docs: List[Document] = []
    for path in data_path.glob("**/*.json"):
        try:
            loaded = JSONLoader(str(path), jq_schema=".", text_content=False).load()
            docs.extend(_tag(d, path, "json") for d in loaded)
            print(f"[DEBUG] Loaded JSON file: {path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {path}: {e}")
    return docs


def _load_images(data_path: Path) -> List[Document]:
    """OCR-based image loading (pytesseract) rather than UnstructuredImageLoader:
    simpler, dependency-light, and debuggable -- images are text-unified into
    the same chunking/embedding path as everything else."""
    docs: List[Document] = []
    for pattern in ("**/*.jpg", "**/*.jpeg", "**/*.png"):
        for path in data_path.glob(pattern):
            try:
                text = pytesseract.image_to_string(Image.open(path))
                if text.strip():
                    doc = Document(page_content=text)
                    docs.append(_tag(doc, path, "image"))
                    print(f"[DEBUG] OCR'd image: {path.name} ({len(text)} chars)")
                else:
                    print(f"[DEBUG] OCR produced no text for image: {path.name}")
            except Exception as e:
                print(f"[ERROR] Failed to OCR image {path}: {e}")
    return docs


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load supported files from the data directory and convert to LangChain
    Document objects. Supported: PDF, TXT, CSV, Excel, Word, JSON, images (OCR).

    Each file type is loaded independently and wrapped in its own try/except
    so one bad file doesn't abort the whole ingestion run.
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    loaders = {
        "pdf": _load_pdfs,
        "text": _load_text_files,
        "csv": _load_csvs,
        "excel": _load_excel,
        "word": _load_word,
        "json": _load_json,
        "image": _load_images,
    }

    documents: List[Any] = []
    counts: dict[str, int] = {}
    for file_type, loader_fn in loaders.items():
        loaded = loader_fn(data_path)
        counts[file_type] = len(loaded)
        documents.extend(loaded)

    print(f"[DEBUG] Ingestion summary: {counts} | total documents: {len(documents)}")
    return documents
