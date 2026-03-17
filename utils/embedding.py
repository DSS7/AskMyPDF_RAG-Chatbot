def chunk_text(text, chunk_size=2):
    """
    Splits text into chunks of `chunk_size` paragraphs.
    Normalizes line breaks first.
    """
    # Replace single line breaks inside paragraphs with space
    text = text.replace("\n", " ")
    # Split by double line breaks or paragraphs (you can also split by ". " if needed)
    paragraphs = text.split(". ")  # split by sentences instead of unreliable line breaks
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = ". ".join(paragraphs[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

from sentence_transformers import SentenceTransformer

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunks):
    """
    Converts a list of text chunks into embeddings (vectors).
    """
    embeddings = embedding_model.encode(chunks)
    return embeddings
