# raknpa_chat_bot.py

import os
import pickle
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- CONFIG ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
PDF_FILES = ["data/PO_Guide_Part-1.pdf"]  # Preloaded PDF(s)

# --- PAGE SETTINGS ---
st.set_page_config(page_title="RAKNPA Chat Bot", layout="wide")

# --- HEADER (Logo + Academy Name) ---
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    st.image("logo.png", width=80)
with header_col2:
    st.markdown("""
    <div style='display: flex; align-items: center; height: 100px;'>
        <h2 style='color: black;'>Rafi Ahmed Kidwai National Postal Academy</h2>
    </div>
    """, unsafe_allow_html=True)

# --- Title Box ---
st.markdown("""
<div style='border: 2px solid #8B0000; padding: 10px; border-radius: 6px; text-align: center;'>
    <h3 style='color: #8B0000; margin: 0;'>RAKNPA Chat Bot</h3>
</div>
""", unsafe_allow_html=True)

# --- LOAD PDFs ---
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
pdf_paths = [os.path.join(data_dir, os.path.basename(p)) for p in PDF_FILES]

index_key = "_".join([os.path.splitext(os.path.basename(p))[0] for p in pdf_paths])
index_path = os.path.join("indexes", f"{index_key}.faiss")
meta_path = os.path.join("indexes", f"{index_key}.pkl")
os.makedirs("indexes", exist_ok=True)

# --- INDEXING ---
if os.path.exists(index_path) and os.path.exists(meta_path):
    with st.spinner("🔄 Loading saved document index..."):
        with open(meta_path, "rb") as f:
            stored_data = pickle.load(f)
            vectordb = FAISS.load_local(index_path, stored_data["embeddings"])
        st.success("✅ Index loaded from disk.")
else:
    with st.spinner("📚 Processing and indexing document(s)..."):
        all_docs = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-3-small"
)
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"embeddings": embeddings}, f)
        st.success("✅ New index created.")

# --- CHAT INTERFACE ---
st.divider()
st.subheader("💬 Ask a question")
query = st.text_input("Type your question here:")

if query:
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key),
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    with st.spinner("🤖 Thinking..."):
        result = qa_chain(query)
        st.markdown("### 🧠 Answer")
        st.markdown(result["result"])

        with st.expander("📄 Source snippets"):
            for doc in result["source_documents"]:
                st.markdown(f"---\n**Snippet:**\n{doc.page_content[:800]}")
