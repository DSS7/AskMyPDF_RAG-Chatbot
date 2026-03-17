import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from embeddings.
    """
    dimension = embeddings[0].shape[0]  # vector size
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

def retrieve_top_chunks(query_embedding, index, chunks, top_k=3):
    """
    Returns the top_k most relevant chunks for a query.
    """
    query_np = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_np, top_k)
    results = [chunks[i] for i in indices[0]]
    return results
