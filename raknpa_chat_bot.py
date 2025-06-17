import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import base64

# ---- UI: Logo and Header ----
st.set_page_config(page_title="RAKNPA Chat Bot", layout="wide")
logo_path = "logo.png"
encoded_logo = base64.b64encode(open(logo_path, "rb").read()).decode()
st.markdown(
    f"""
    <div style="display: flex; align-items: center; border-bottom: 3px solid darkred; padding: 10px; background-color: white;">
        <img src="data:image/png;base64,{encoded_logo}" style="height: 60px; margin-right: 15px;">
        <h1 style="color: black; margin: 0;">Rafi Ahmed Kidwai National Postal Academy</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style='color: darkred;'>Ask Your Questions Below</h2>", unsafe_allow_html=True)

# ---- Load & Process PDF ----
pdf_dir = "data"
all_text = ""
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(os.path.join(pdf_dir, filename))
        for page in pdf_reader.pages:
            all_text += page.extract_text() or ""

# ---- Text Split ----
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(all_text)

# ---- Embeddings (Local) ----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_texts(chunks, embedding=embeddings)

# ---- Streamlit Chat UI ----
query = st.text_input("Enter your question:", "")
if query:
    docs = vectordb.similarity_search(query)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.markdown(f"**Answer:** {response}")
