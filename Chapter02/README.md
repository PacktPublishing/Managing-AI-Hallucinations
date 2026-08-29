# Chapter 2 - Preventing Hallucinations

This directory contains the code and supporting files for the local retrieval-augmented generation (RAG) example used in Chapter 2. The example ingests local PDF and text files into ChromaDB, retrieves relevant passages, and asks an OpenAI model to answer from that context.

## Prerequisites

- Python 3.11 or newer. The source project does not declare an exact Python version; the copied project was verified with Python 3.13.1.
- `pip`
- An OpenAI API key. Ingestion creates OpenAI embeddings, and querying calls an OpenAI chat model, so both operations use the API and may incur charges.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env`. Edit `.env` and replace the placeholder with your key:

```dotenv
OPENAI_API_KEY=your-openai-api-key-here
```

Do not commit `.env`. It is excluded by this chapter's `.gitignore`.

## Run the example

Run all commands from the `Chapter02` directory.

Ingest every supported file in `data/`:

```powershell
python src/ingest.py
```

After ingestion has created `chroma_db/`, run the example queries:

```powershell
python src/query.py
```

The combined reader-facing demonstration ingests the files and asks two questions:

```powershell
python test_rag.py
```

The source README calls `python test_rag.py` a test. It is an end-to-end demonstration rather than an isolated unit test: it creates embeddings, writes a local ChromaDB store, and makes live model requests. It therefore requires a valid API key and network access.

## Files

- `data/DataMind_FAQ_EN.pdf` contains the sample DataMind FAQ used for retrieval.
- `data/additional_info.txt` contains additional DataMind support and account information.
- `src/ingest.py` loads PDF, text, and Markdown files, splits them into chunks, creates embeddings, persists them in ChromaDB, and tracks file hashes to avoid unnecessary re-ingestion.
- `src/query.py` loads the persisted vector store, retrieves relevant chunks, and generates a grounded answer with source metadata.
- `src/rag.py` provides the `RAGSystem` facade that combines ingestion and querying.
- `test_rag.py` runs the complete ingestion-and-query demonstration used in the chapter.

## Book section

This project supports the local RAG practical example in Chapter 2, **Preventing Hallucinations**. It demonstrates grounding model output in reader-controlled documents instead of relying only on the model's internal knowledge.
