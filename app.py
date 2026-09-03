"""End-to-end smoke test for the core pipeline: sync (incremental
ingest/update/remove) -> retrieve. Proves the non-agentic pipeline works
before LangGraph orchestration is layered on top (Phase 3).

Uses sync.py rather than a blind full re-ingest, so re-running this is cheap
-- only new/modified/removed files under DATA_DIR are processed."""

from rag_learn import config
from rag_learn.embedding import EmbeddingPipeline
from rag_learn.search import Retriever
from rag_learn.sync import sync
from rag_learn.vectorstore import VectorStore

if __name__ == "__main__":
    sync()

    pipeline = EmbeddingPipeline()
    store = VectorStore()
    retriever = Retriever(store, pipeline)

    results = retriever.retrieve("What is Deepak's experience?", top_k=config.TOP_K)
    for r in results:
        print(f"\n[{r['rank']}] score={r['score']:.3f} source={r['metadata']['source_file']}")
        print(r["content"][:200])
