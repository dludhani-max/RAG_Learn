"""Semantic Q&A cache: reuses a past answer for a new question that's
close enough in meaning, instead of re-running retrieval + LLM calls.

Shared across all users (a ChromaDB collection, not Streamlit session
state) -- if anyone already asked an equivalent question, everyone
benefits. Cleared entirely by sync.py whenever the document set changes,
since a fine-grained "which cached answers depend on which source
documents" invalidation isn't worth the complexity at this scale, and a
stale answer being served silently would be worse than a cache miss.
"""

import hashlib
import json
import os
from typing import Any, Optional

import chromadb

from rag_learn import config


class QACache:
    def __init__(
        self,
        collection_name: str = config.CACHE_COLLECTION_NAME,
        persistent_directory: str = config.VECTOR_STORE_DIR,
        similarity_threshold: float = config.CACHE_SIMILARITY_THRESHOLD,
    ):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.similarity_threshold = similarity_threshold
        os.makedirs(self.persistent_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persistent_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Cached question -> answer pairs", "hnsw:space": "cosine"},
        )

    def lookup(self, question: str, question_embedding) -> Optional[dict[str, Any]]:
        if self.collection.count() == 0:
            return None

        results = self.collection.query(query_embeddings=[question_embedding.tolist()], n_results=1)
        if not results["documents"] or not results["documents"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance
        if similarity < self.similarity_threshold:
            return None

        metadata = results["metadatas"][0][0]
        return {
            "cached_question": results["documents"][0][0],
            "generation": metadata["generation"],
            "sources": json.loads(metadata["sources_json"]),
            "similarity": similarity,
        }

    def store(self, question: str, question_embedding, generation: str, sources: list[dict[str, Any]]):
        doc_id = f"qa_{hashlib.sha256(question.encode()).hexdigest()[:16]}"
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[question_embedding.tolist()],
            documents=[question],
            metadatas=[{"generation": generation, "sources_json": json.dumps(sources)}],
        )

    def clear(self):
        """Wipe the whole cache -- called by sync.py after any document
        change, since a stale cached answer served silently is worse than a
        cache miss."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # nothing to delete yet
        self.collection = self._get_or_create_collection()
        print(f"[CACHE] Cleared Q&A cache (collection: {self.collection_name})")
