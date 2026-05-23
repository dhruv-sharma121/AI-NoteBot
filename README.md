# 📘 RAGMind AI

### AI-Powered Multi-PDF Study Assistant using Retrieval-Augmented Generation (RAG)

RAGMind AI is an intelligent AI-powered study assistant that allows users to upload multiple PDF documents and ask questions from them using Retrieval-Augmented Generation (RAG).

The system uses semantic search, vector embeddings, and Large Language Models (LLMs) to provide accurate answers, summaries, and viva preparation content from uploaded PDFs.

---

# 🚀 Features

✅ Multi-PDF Upload Support  
✅ Ask Questions from PDFs  
✅ AI Generated Summaries  
✅ Viva Question Generation  
✅ Semantic Search using Vector Embeddings  
✅ FAISS Vector Database  
✅ Groq LLM Integration  
✅ Fast Retrieval-Augmented Generation (RAG) Pipeline  

---

# 🧠 Architecture

```text
PDFs
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embeddings Generation
   ↓
FAISS Vector Store
   ↓
Semantic Search
   ↓
Top Relevant Chunks
   ↓
Groq LLM
   ↓
AI Generated Answer

```

# 🛠 Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Backend Logic |
| Streamlit | Frontend UI |
| LangChain | AI Workflow |
| FAISS | Vector Database |
| HuggingFace Embeddings | Semantic Embeddings |
| Groq LLM | Answer Generation |
| PyPDF2 | PDF Text Extraction |



# 📸 Application Features

## 🔹 Ask Questions

Users can ask questions from uploaded PDFs and receive AI-generated answers based only on document content.

---

## 🔹 Generate Summary

Automatically generates concise summaries from uploaded study material.

---

## 🔹 Generate Viva Questions

Creates important viva/interview questions from uploaded PDFs.

# 🧩 How RAGMind AI Works
```text
PDFs are uploaded by the user
Text is extracted from documents
Text is split into smaller chunks
Chunks are converted into embeddings
Embeddings are stored in FAISS vector database
User query is converted into embedding
Semantic similarity search retrieves relevant chunks
Retrieved chunks are passed to Groq LLM
LLM generates final answer
```
# 📈 Future Improvements
```text
Conversational Chat Memory
Source Citation with Page Numbers
OCR Support for Scanned PDFs
Voice Interaction
Flashcard Generator
MCQ Generation
ChatGPT-like UI
Multi-language Support
PDF Highlighting
```
# 🖼 Sample Architecture
![Architecture](architecture.png)
