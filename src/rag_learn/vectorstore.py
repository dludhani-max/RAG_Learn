import hashlib
import json
import os
from collections import defaultdict
from typing import Any, List

import chromadb
import numpy as np

from rag_learn import config


def _stable_id(source: str, chunk_index: int) -> str:
    """Deterministic id from source path + chunk index, so re-ingesting the
    same file doesn't create duplicate rows (unlike a random uuid per add)."""
    digest = hashlib.sha256(f"{source}::{chunk_index}".encode()).hexdigest()[:16]
    return f"doc_{digest}"


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        persistent_directory: str = config.VECTOR_STORE_DIR,
    ):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persistent_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistent_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # Explicitly cosine: ChromaDB defaults to L2 (Euclidean)
                # distance, which is unbounded and breaks the
                # `similarity = 1 - distance` conversion used in search.py
                # (that formula only holds for cosine distance, bounded [0,2]).
                metadata={"description": "Document embeddings for RAG", "hnsw:space": "cosine"},
            )
            print(f"[INFO] VectorStore initialized. Collection: {self.collection_name}")
            print(f"[INFO] Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"[ERROR] Error initializing vector store: {e}")
            raise

    def count(self) -> int:
        return self.collection.count()

    def delete(self, ids: List[str]):
        """Remove chunks by id -- used by sync.py to clear a file's old
        chunks before re-adding it (modified file) or to drop them entirely
        (deleted file)."""
        if not ids:
            return
        self.collection.delete(ids=ids)
        print(f"[INFO] Deleted {len(ids)} chunks. Vector store now has {self.collection.count()} documents")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> List[str]:
        """Add documents and their embeddings to the vector store.

        Uses a stable id (hash of source + the chunk's index *within its own
        source file*) per chunk, so re-running ingestion on the same files
        updates existing rows instead of duplicating them -- independent of
        what other files happen to be in the same batch (chunk index is
        computed per-source here, not as a position in the whole batch,
        since a global batch index would shift for unrelated files whenever
        the batch composition changes).

        Returns the ids that were written, so callers (sync.py) can record
        them for later deletion.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"[INFO] Adding {len(documents)} documents to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []
        per_source_index: dict[str, int] = defaultdict(int)

        for doc, embedding in zip(documents, embeddings):
            source = str(doc.metadata.get("source", doc.metadata.get("source_file", "")))
            source_chunk_index = per_source_index[source]
            per_source_index[source] += 1
            ids.append(_stable_id(source, source_chunk_index))

            page = doc.metadata.get("page")
            try:
                page = int(page) if page is not None else -1
            except (TypeError, ValueError):
                page = -1

            metadatas.append(
                {
                    "source_file": str(doc.metadata.get("source_file", "")),
                    "file_type": str(doc.metadata.get("file_type", "")),
                    "page": page,
                    "doc_index": source_chunk_index,
                    "content_length": len(doc.page_content),
                    # Chroma metadata values must be scalars, so a list of
                    # extracted diagram image paths is JSON-encoded here and
                    # decoded back out in graph.py when building sources.
                    "images_json": json.dumps(doc.metadata.get("images", [])),
                    # Set by classifier.classify_and_tag before chunking
                    # (Phase 2.5) -- Phase 3b's SQL/page-index retrieval
                    # paths read this to decide how to serve a given chunk.
                    # Defaults to "vector" so pre-Phase-2.5 chunks (or any
                    # document type the classifier doesn't tag) fall back to
                    # today's classic embedding search rather than an
                    # unrecognized/missing route.
                    "routing": str(doc.metadata.get("routing", "vector")),
                }
            )
            documents_text.append(str(doc.page_content))
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"[INFO] Vector store now has {self.collection.count()} documents")
            return ids
        except Exception as e:
            print(f"[ERROR] Error adding documents to vector store: {e}")
            raise
