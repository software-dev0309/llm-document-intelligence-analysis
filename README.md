# LLM-Document Intelligent Analysis

A full-stack AI-powered system for analyzing documents and intelligently answering user questions. This application extracts content from uploaded PDF files, chunks and embeds the text, stores it in a vector database, and uses a Retrieval-Augmented Generation (RAG) pipeline with large language models to generate grounded answers.

## Features

- Upload and process scanned or digital PDF files.
- Intelligent text chunking and embedding.
- Semantic search using FAISS.
- Question answering with Hugging Face LLMs (Gemma, LLaMA 3).
- Streamlit-based user interface for ease of use.
- Model quantization and efficient GPU/CPU execution via `bitsandbytes`.

---
