from typing import Any, Dict, List, Optional

from rag_learn.embedding import EmbeddingPipeline
from rag_learn.vectorstore import VectorStore


class Retriever:
    """Handles query embedding + retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_pipeline: EmbeddingPipeline):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        source_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """source_file: when given, scope vector search to chunks from that
        one source file (Phase 3b's "search this document" mode) instead of
        the whole collection -- e.g. for a vector-routed document a future
        UI lets the user pick by name, same as it would for a SQL table or
        page-index tree."""
        print(f"[INFO] Retrieving documents for query: '{query}' (top_k={top_k}, score_threshold={score_threshold})")

        query_embedding = self.embedding_pipeline.model.encode([query])[0]
        where = {"source_file": source_file} if source_file else None

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
            )
        except Exception as e:
            print(f"[ERROR] Error during retrieval: {e}")
            return []

        retrieved_docs: List[Dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for rank, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances), start=1
            ):
                # ChromaDB's default space is cosine distance -> similarity = 1 - distance
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "score": similarity_score,
                            "distance": distance,
                            "rank": rank,
                        }
                    )

        print(f"[INFO] Retrieved {len(retrieved_docs)} documents (after score filtering)")
        return retrieved_docs
