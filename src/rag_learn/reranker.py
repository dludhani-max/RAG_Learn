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

        scored = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)

        # Greedily fill top_k while capping chunks per source_file, so one
        # dominant document can't crowd out other relevant ones. If capping
        # would leave slots unfilled (fewer than top_k distinct-enough
        # candidates), relax the cap on a second pass over the leftovers
        # rather than returning fewer than top_k results.
        per_source_count: dict[str, int] = {}
        selected: list[tuple[dict[str, Any], float]] = []
        leftover: list[tuple[dict[str, Any], float]] = []
        for doc, score in scored:
            source = doc.get("metadata", {}).get("source_file", "")
            if per_source_count.get(source, 0) < config.MAX_PER_SOURCE:
                selected.append((doc, score))
                per_source_count[source] = per_source_count.get(source, 0) + 1
            else:
                leftover.append((doc, score))
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            selected.extend(leftover[: top_k - len(selected)])

        results = []
        for rank, (doc, score) in enumerate(selected, start=1):
            updated = dict(doc)
            updated["vector_score"] = doc.get("score")
            updated["score"] = float(score)
            updated["rank"] = rank
            results.append(updated)
        return results
