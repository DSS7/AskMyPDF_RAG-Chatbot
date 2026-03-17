import streamlit as st
from utils.pdf_loader import load_pdf
from utils.embedding import chunk_text, get_embeddings
from utils.retrieval import build_faiss_index, retrieve_top_chunks
from utils.llm import generate_answer
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AskMyPDF RAG Chatbot", layout="wide")
st.title("AskMyPDF RAG Chatbot")

# -------------------------------
# Load embedding model once
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -------------------------------
# Process PDF: chunking, embeddings, FAISS
# -------------------------------
@st.cache_data(show_spinner=False)
def process_pdf(uploaded_file):
    text = load_pdf(uploaded_file)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)
    return chunks, embeddings, index

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Process PDF once
    chunks, embeddings, index = process_pdf(uploaded_file)
    st.success(f"PDF processed! Total Chunks: {len(chunks)}")
    
    # -------------------------------
    # User query input
    # -------------------------------
    query = st.text_input("Ask a question about the PDF:")

    if query:
        # 1️⃣ Generate query embedding
        query_embedding = embedding_model.encode([query])[0]
        
        # 2️⃣ Retrieve top relevant chunks
        top_chunks = retrieve_top_chunks(query_embedding, index, chunks)
        
        st.subheader("Top Relevant Chunks:")
        for i, chunk in enumerate(top_chunks):
            st.write(f"Chunk {i+1}: {chunk}")
        
        # 3️⃣ Generate answer from LLM
        answer = generate_answer(top_chunks, query)
        st.subheader("Answer from Chatbot:")
        st.write(answer)
