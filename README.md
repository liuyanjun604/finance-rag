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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM & Embeddings | Qwen via SiliconFlow API |
