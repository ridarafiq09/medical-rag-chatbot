# 🩺 Medical RAG Chatbot

*A Retrieval-Augmented Medical Assistant using FAISS, HuggingFace Embeddings & Groq Llama-3.1*

---

# Overview

This project implements an end-to-end **Medical Retrieval-Augmented Generation (RAG)** system using:

* **FAISS** for vector search
* **HuggingFace MiniLM embeddings**
* **Groq Llama-3.1-8B-Instant** for lightning-fast inference
* **Streamlit** for an interactive medical chatbot UI

The system loads medical PDF documents, chunks them, embeds them, stores them in a FAISS vector database, retrieves relevant context, and generates grounded medical answers with source-document transparency.

---

## Features

### **RAG Pipeline**

* Load and process medical PDFs
* Chunk text using `RecursiveCharacterTextSplitter`
* Create embeddings with `all-MiniLM-L6-v2`
* Store vectors in a **local FAISS database**
* Retrieve top-k relevant chunks during queries

### **LLM Integration**

* Uses **Groq’s Llama-3.1-8B-Instant** (extremely fast inference)
* Custom medical-safe prompt
* Accurate, grounded responses

### **Streamlit Chatbot**

* Clean conversational UI
* Greeting detection
* Persistent chat history
* Shows retrieved **source documents**
* Error handling built-in

---

## 📁 Project Structure

```
project/
│
├── data/                      # Medical PDFs
├── vectorstore/
│   └── db_faiss/              # FAISS vector database
│
├── phase1_create_vectorstore.py  # PDF → Chunks → Embeddings → FAISS
├── phase2_test_rag.py            # Test RAG pipeline from terminal
├── Chatbot.py                        # Streamlit RAG chatbot
│
├── requirements.txt
└── README.md
```

---

## How It Works

### **Phase 1 — Build Vector Store**

1. Load PDFs using `PyPDFLoader`
2. Chunk text with overlap
3. Generate embeddings
4. Store vectors in FAISS
5. Save database locally

### **Phase 2 — RAG Chain**

1. Load FAISS vector DB
2. Load Groq LLM
3. Create retrieval pipeline
4. Format context
5. Answer queries with grounded context

### **Phase 3 — Streamlit Chatbot**

1. User sends query
2. System checks for greeting
3. If not greeting → run RAG
4. Display LLM answer
5. Show source documents used

---

## 🛠️ Technologies Used

### **Core**

* Python
* LangChain
* Streamlit

### **AI & NLP**

* Groq Llama-3.1
* HuggingFace MiniLM embeddings
* FAISS vector database

### **Utilities**

* dotenv
* PyPDFLoader
* RecursiveCharacterTextSplitter

---
