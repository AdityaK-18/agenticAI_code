from sentence_transformers import SentenceTransformer

# ✅ Load the embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Function to generate embeddings
def generate_embedding(text):
    return embed_model.encode(text)
