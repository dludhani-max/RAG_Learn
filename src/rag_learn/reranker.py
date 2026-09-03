from typing import Any

from sentence_transformers import CrossEncoder

from rag_learn import config


class Reranker:
    """Cross-encoder reranker: scores each (query, chunk) pair jointly,
    which is a more accurate relevance signal than embedding similarity
    (computed independently for query and chunk) -- used to reorder a wider
    candidate set down to the final top_k before grading/generation."""

    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Loaded reranker model: {model_name}")

    def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not documents:
            return []

        pairs = [(query, d["content"]) for d in documents]
        scores = self.model.predict(pairs)

        reranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

        results = []
        for rank, (doc, score) in enumerate(reranked, start=1):
            updated = dict(doc)
            updated["vector_score"] = doc.get("score")
            updated["score"] = float(score)
            updated["rank"] = rank
            results.append(updated)
        return results
