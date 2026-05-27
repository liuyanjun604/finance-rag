# Finance RAG — Financial Document Q&A System

A Retrieval-Augmented Generation (RAG) application that allows users to upload financial PDF documents and ask questions in natural language.

Built with real-world financial domain knowledge from 3 years of experience at Citi.

## Features

- Upload PDF documents and automatically parse, chunk, and index them
- Natural language Q&A powered by LLM, answers grounded strictly in document content
- Streaming response for real-time output
- Multi-turn conversation memory
- REST API backend with FastAPI
- Interactive web UI with Streamlit
- Tool Calling support: real-time stock price queries via function calling

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM & Embeddings | Qwen via SiliconFlow API |
| RAG Framework | LangChain |
| Vector Database | Chroma |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| PDF Parsing | PyPDF2 |

## Architecture

    User uploads PDF
          ↓
    PyPDF2 parses → chunks text (RecursiveCharacterTextSplitter)
          ↓
    Embedding API converts chunks to vectors
          ↓
    Chroma stores vectors + page metadata
          ↓
    User asks a question
          ↓
    Question → vector → retrieve top-k chunks
          ↓
    Chunks + history → LLM prompt → streaming response

## Getting Started

**Prerequisites:** Python 3.10+, SiliconFlow API key

**Install:**

    git clone https://github.com/liuyanjun604/finance-rag.git
    cd finance-rag
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

**Configure:** Create a `.env` file:

    SILICONFLOW_API_KEY=your_api_key_here

**Run:**

    # Terminal 1 - Backend
    uvicorn src.api:app --reload

    # Terminal 2 - Frontend
    streamlit run src/ui.py

Open `http://localhost:8501` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF file for processing |
| POST | `/ask` | Ask a question (returns JSON) |
| POST | `/ask-stream` | Ask a question (streaming response) |

## Project Structure

    finance-rag/
    ├── src/
    │   ├── api.py        # FastAPI backend
    │   ├── ui.py         # Streamlit frontend
    │   └── config.py     # Configuration parameters
    ├── .gitignore
    ├── requirements.txt
    └── README.md
