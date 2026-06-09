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
- LangGraph Agent: autonomous tool-calling loop with state management
- Hybrid search: combines BM25 keyword search and vector similarity, fused with Reciprocal Rank Fusion (RRF) for better retrieval

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM & Embeddings | Qwen via SiliconFlow API |
| RAG Framework | LangChain |
| Agent Orchestration | LangGraph |
| Vector Database | Chroma |
| Keyword Search | BM25 (rank-bm25) |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Stock Data | yfinance |

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
    Hybrid search: BM25 keyword + vector similarity → RRF fusion → top-k chunks
          ↓
    Chunks + history → LangGraph Agent
          ↓
    Agent decides: answer from docs, or call tools (e.g. stock price)
          ↓
    Streaming response

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

## Configuration

All tunable parameters live in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Chat model used for answers and tool calling |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Model used to embed chunks and queries |
| `BASE_URL` | `https://api.siliconflow.com/v1` | OpenAI-compatible API endpoint |
| `CHUNK_SIZE` | `1000` | Max characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping characters between adjacent chunks |
| `N_RESULTS` | `5` | Top-k results returned per retrieval path (BM25 / vector) |

The hybrid retrieval strategy is toggled by `USE_RRF` in `src/hybrid_search.py` (`True` = RRF fusion, `False` = simple merge-and-dedup).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF file for parsing, chunking, and indexing |
| POST | `/ask-stream` | Ask a question (streaming response, with conversation history) |

## Project Structure

    finance-rag/
    ├── src/
    │   ├── api.py            # FastAPI backend: /upload and /ask-stream endpoints
    │   ├── ui.py             # Streamlit frontend: chat UI with file upload
    │   ├── agent.py          # LangGraph Agent: autonomous tool-calling loop
    │   ├── hybrid_search.py  # Hybrid retrieval: BM25 + vector search, RRF fusion
    │   ├── tool_calling.py   # Tools: real-time stock price query (yfinance)
    │   └── config.py         # Configuration parameters
    ├── benchmarks/
    │   └── bench_bm25_rrf.py # BM25 rebuild-vs-reuse latency & RRF-vs-merge ordering
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Benchmarks

`benchmarks/bench_bm25_rrf.py` measures retrieval performance on the real
annual-report corpus:

- BM25 latency: rebuilding the index per query vs. reusing a pre-built index
- Fusion ordering: RRF vs. simple merge-and-dedup

Run it from the project root (requires a local PDF; the sample report is
git-ignored):

    python benchmarks/bench_bm25_rrf.py
