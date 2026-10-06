# Multimodal RAG

A production-ready starter project for building a multimodal retrieval-augmented generation pipeline over PDFs and other document sources.

## Overview

This repository demonstrates a simple architecture for:

- ingesting PDF text, tables, and images
- embedding text and image content
- storing vectors in ChromaDB
- retrieving top-k relevant chunks
- reranking results
- generating answers with GPT-4o
- exposing a FastAPI API and optional Streamlit UI

## Project Structure

```text
multimodal-rag/
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── embedder.py
│   └── indexer.py
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py
│   └── reranker.py
├── generation/
│   ├── __init__.py
│   └── generator.py
├── api/
│   ├── __init__.py
│   └── main.py
├── ui/
│   ├── __init__.py
│   └── app.py
├── tests/
│   ├── __init__.py
│   └── test_placeholder.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Core Workflow

1. Parse PDF content into chunks, tables, and image references.
2. Generate embeddings for text and image payloads.
3. Store embeddings in ChromaDB.
4. Query the vector database for similar content.
5. Optionally rerank retrieval results.
6. Pass the most relevant context to GPT-4o for answer generation.

## Quick Start

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI app:

```bash
uvicorn api.main:app --reload
```

4. Start the Streamlit UI:

```bash
streamlit run ui/app.py
```

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_key_here
```

## Notes

This project is intentionally structured as a clean starter template. It is designed to be extended with real multi-vector indexing, OCR, table parsing, and production-grade evaluation pipelines.

## License

This project is provided as a starter scaffold for learning and experimentation.
