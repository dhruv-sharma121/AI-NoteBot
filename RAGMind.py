import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# QA Chain
from langchain_classic.chains.question_answering import load_qa_chain

# Prompt Template
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------
# API KEY
# ---------------------------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="RAGMind",
    page_icon="📘",
    layout="wide"
)

st.title("📘 RAGMind - AI Powered Multi-PDF Study Assistant")

st.write(
    "Upload multiple PDFs and ask questions from all documents together."
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("📂 Upload PDFs")

    files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

# ---------------------------------------------------
# PROCESS MULTIPLE PDFs
# ---------------------------------------------------

if files:

    all_text = ""

    # READ ALL PDFs
    for file in files:

        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                all_text += text

    # ---------------------------------------------------
    # TEXT SPLITTING
    # ---------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(all_text)

    # ---------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------
    # VECTOR STORE
    # ---------------------------------------------------

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    # ---------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------

    option = st.selectbox(
        "Choose Feature",
        [
            "Ask Questions",
            "Generate Summary",
            "Generate Viva Questions"
        ]
    )

    # ---------------------------------------------------
    # LLM
    # ---------------------------------------------------

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    # ===================================================
    # 1. ASK QUESTIONS
    # ===================================================

    if option == "Ask Questions":

        user_query = st.text_input(
            "Ask a question from your PDFs:"
        )

        if user_query:
            with st.spinner("Generating Answer..."):
                docs = vector_store.similarity_search(
                    user_query,
                    k=5
                )

                # STRICT PDF PROMPT
                prompt_template = """
                You are an AI Study Assistant.

                Answer ONLY using the provided PDF context.

                If answer is not available in the PDFs,
                say:
                "Answer not found in uploaded PDFs."

                Context:
                {context}

                Question:
                {question}

                Answer:
                """

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                chain = load_qa_chain(
                    llm,
                    chain_type="stuff",
                    prompt=PROMPT
                )

                response = chain.run(
                    input_documents=docs,
                    question=user_query
                )

                st.subheader("📌 Answer")
                st.write(response)



    # ===================================================
    # 2. GENERATE SUMMARY
    # ===================================================

    elif option == "Generate Summary":

            with st.spinner("Generating Summary..."):

                docs = vector_store.similarity_search(
                    "Important content",
                    k=8
                )

                prompt_template = """
                You are an AI PDF Summarizer.

                Generate a detailed summary
                using ONLY the provided PDF context.

                Context:
                {context}

                Question:
                {question}

                Summary:
                """

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                chain = load_qa_chain(
                    llm,
                    chain_type="stuff",
                    prompt=PROMPT
                )

                response = chain.run(
                    input_documents=docs,
                    question="Generate summary"
                )

                st.subheader("📄 Summary")
                st.write(response)

    # ===================================================
    # 3. VIVA QUESTIONS
    # ===================================================

    elif option == "Generate Viva Questions":

        # if st.button("Generate Viva Questions"):

            with st.spinner("Generating Viva Questions..."):

                docs = vector_store.similarity_search(
                    "Important viva questions",
                    k=8
                )

                prompt_template = """
                You are an AI Viva Preparation Assistant.

                Generate 15 important viva questions
                using ONLY the provided PDF context.

                Rules:
                - Questions should be important
                - Avoid duplicates
                - Do not provide answers

                Context:
                {context}

                Question:
                {question}

                Viva Questions:
                """

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                chain = load_qa_chain(
                    llm,
                    chain_type="stuff",
                    prompt=PROMPT
                )

                response = chain.run(
                    input_documents=docs,
                    question="Generate viva questions"
                )

                st.subheader("🎓 Viva Questions")
                st.write(response)

else:
    st.info("Please upload at least one PDF file.")
