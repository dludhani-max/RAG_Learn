from rag_learn.data_loader import load_all_documents
from rag_learn.embedding import EmbeddingPipeline



# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    docs = load_all_documents("data")
    chunks=EmbeddingPipeline().chunk_documents(docs)
    chunksvectors=EmbeddingPipeline().embed_chunks(chunks)    
    print(chunksvectors)