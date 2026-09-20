# ResearchLens

ResearchLens is a real-time research paper question-answering application built using Retrieval-Augmented Generation (RAG).

It allows users to upload a research paper PDF and ask questions about its content. The system retrieves relevant sections from the document and uses a local Large Language Model (LLM) to generate an answer based only on the retrieved PDF content.

## Features

- Upload research paper PDFs
- Semantic search using sentence embeddings
- Retrieval-Augmented Generation (RAG)
- FAISS vector search
- Local Qwen LLM
- Source page references
- Document-based answers
- Technical term retrieval

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Llama.cpp
- Qwen 2.5
- NumPy

## System Architecture

```text
Research Paper PDF
        |
        v
  PDF Processing
        |
        v
     Chunking
        |
        v
  Text Embeddings
        |
        v
 FAISS Vector Search
        |
        v
Relevant PDF Chunks
        |
        v
  Local Qwen LLM
        |
        v
      Answer
        |
        v
 Source Page Numbers