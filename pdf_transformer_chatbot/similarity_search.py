import faiss
import numpy as np
from embedding_model import generate_embedding

# ✅ FAISS Index
class SimilaritySearch:
    def __init__(self):
        self.index = None
        self.chunks = []

    # ✅ Build FAISS index from text chunks
    def build_index(self, text_chunks):
        self.chunks = text_chunks
        chunk_embeddings = np.array([generate_embedding(chunk) for chunk in text_chunks])

        self.index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        self.index.add(chunk_embeddings)

    # ✅ Retrieve the most relevant text chunk
    def retrieve(self, query):
        query_embedding = generate_embedding([query]).reshape(1, -1)
        _, index_list = self.index.search(query_embedding, 1)
        return self.chunks[index_list[0][0]]

# ✅ Global instance of SimilaritySearch
similarity_search = SimilaritySearch()
