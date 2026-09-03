import hashlib
import os
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

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store.

        Uses a stable id (hash of source + chunk index) per chunk so
        re-running ingestion on the same files updates existing rows
        instead of duplicating them.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"[INFO] Adding {len(documents)} documents to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            source = str(doc.metadata.get("source", doc.metadata.get("source_file", "")))
            ids.append(_stable_id(source, i))

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
                    "doc_index": i,
                    "content_length": len(doc.page_content),
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
        except Exception as e:
            print(f"[ERROR] Error adding documents to vector store: {e}")
            raise
