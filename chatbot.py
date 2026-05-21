import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Free Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# GROQ LLM
from langchain_groq import ChatGroq

from langchain_classic.chains.question_answering import load_qa_chain

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI NoteBot",
    page_icon="📘"
)

st.header("📘Smart AI NoteBot using Groq + Llama3 \n Upload a PDF file to ask questions")

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.title("Upload Your Notes")



    file = st.file_uploader(
        "Upload PDF File",
        type="pdf"
    )

# -----------------------------------
# PROCESS PDF
# -----------------------------------

if file is not None:

    # Read PDF
    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    # -----------------------------------
    # SPLIT TEXT
    # -----------------------------------

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = splitter.split_text(text)

    # -----------------------------------
    # EMBEDDINGS
    # -----------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------------
    # VECTOR STORE
    # -----------------------------------

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    # -----------------------------------
    # USER QUERY
    # -----------------------------------

    user_query = st.text_input(
        "Ask a question from your notes:"
    )

    if user_query:

        with st.spinner("Generating Answer..."):

            # Similarity Search
            matching_chunks = vector_store.similarity_search(
                user_query
            )

            # -----------------------------------
            # GROQ MODEL
            # -----------------------------------

            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.1-8b-instant",
                temperature=0
            )

            # -----------------------------------
            # QA CHAIN
            # -----------------------------------

            chain = load_qa_chain(
                llm,
                chain_type="stuff"
            )

            # -----------------------------------
            # RESPONSE
            # -----------------------------------

            response = chain.run(
                input_documents=matching_chunks,
                question=user_query
            )

            # -----------------------------------
            # DISPLAY RESPONSE
            # -----------------------------------

            st.subheader("Answer")
            st.write(response)
