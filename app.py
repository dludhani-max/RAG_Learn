"""End-to-end smoke test for the core pipeline: load -> chunk -> embed ->
store -> retrieve one query. Proves the non-agentic pipeline works before
LangGraph orchestration is layered on top (Phase 3)."""

from rag_learn import config
from rag_learn.data_loader import load_all_documents
from rag_learn.embedding import EmbeddingPipeline
from rag_learn.search import Retriever
from rag_learn.vectorstore import VectorStore

if __name__ == "__main__":
    docs = load_all_documents(config.DATA_DIR)

    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    embeddings = pipeline.embed_chunks(chunks)

    store = VectorStore()
    store.add_documents(chunks, embeddings)

    retriever = Retriever(store, pipeline)
    results = retriever.retrieve("What is Deepak's experience?", top_k=config.TOP_K)
    for r in results:
        print(f"\n[{r['rank']}] score={r['score']:.3f} source={r['metadata']['source_file']}")
        print(r["content"][:200])
