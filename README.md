# AI Knowledge Assistant

A local-first RAG (Retrieval-Augmented Generation) pipeline for PDF document Q&A: upload PDFs, ask questions in plain English, get answers with **page-level source citations** — every answer tells you exactly which document and page it came from. No API keys, no cloud cost — everything runs on your machine via [Ollama](https://ollama.com).

Modeled after [Mario Duerson's AI Knowledge Assistant](https://github.com/Zo-hund/ai-portfolio/tree/main/1-knowledge-assistant), simplified into a single lean Python service.

## Features

- **PDF ingestion** — parses PDFs page-by-page and splits them into overlapping, page-tagged chunks.
- **Local embeddings + generation** — no OpenAI/Anthropic API key required; everything runs through a local Ollama server.
- **Page-level citations** — answers are constrained to retrieved context and cite `(document, page)` for every fact.
- **Web UI + REST API** — a minimal single-file HTML/JS frontend, plus a documented FastAPI backend you can call directly.
- **Programmatic usage** — every capability is also callable as a plain Python function (see [`examples/quickstart.py`](examples/quickstart.py)), no HTTP server required.
- **Bonus: Matryoshka embedding benchmark** — a standalone experiment (`experiments/matryoshka_benchmark.py`) measuring how much embedding storage you can save by truncating dimensions (768→512→256→128) before it hurts retrieval quality.

## Tech stack

- **FastAPI** — backend + minimal HTML/JS UI (no frontend framework, no build step)
- **ChromaDB** — local, on-disk vector store
- **Ollama** — local embedding model (`nomic-embed-text`) + local generation model (`qwen2.5:3b`)
- **pypdf** — page-aware PDF text extraction

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- `pip install -r requirements.txt`

## Setup

1. Install [Ollama](https://ollama.com) and pull the required models:
   ```
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) copy `.env.example` to `.env` to override any defaults:
   ```
   cp .env.example .env
   ```

4. Run the app:
   ```
   uvicorn app.main:app --reload
   ```

5. Open http://localhost:8000, upload a PDF, and ask questions about it.

## Quickstart

**Via the web UI** — open http://localhost:8000, upload a PDF, type a question.

**Via curl:**
```bash
curl -X POST http://localhost:8000/upload -F "file=@your_document.pdf"

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

**Programmatically, no server needed:**
```bash
python examples/quickstart.py your_document.pdf "What does this document say about X?"
```

## API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | — | The web UI |
| `POST` | `/upload` | multipart form, field `file` (PDF) | `{"filename": str, "chunks_indexed": int}` |
| `POST` | `/ask` | `{"question": str}` | `{"answer": str, "sources": [{"source": str, "page": int}, ...]}` |
| `GET` | `/sources` | — | `{"sources": [str, ...]}` — list of indexed documents |

## Data formats

Every chunk stored in ChromaDB carries page-level metadata so answers can cite their source:
```json
{
  "text": "chunk text extracted from the PDF page",
  "source": "your_document.pdf",
  "page": 3
}
```

## Programmatic examples

```python
from app.ingest import ingest_pdf
from app.qa import answer_question

# Ingest: parse, chunk, embed, and store a PDF
chunk_count = ingest_pdf("your_document.pdf", "your_document.pdf")

# Ask: embed the question, retrieve top-k chunks, generate a cited answer
result = answer_question("What does this document say about X?")
print(result["answer"])
print(result["sources"])  # [{"source": "your_document.pdf", "page": 3}, ...]
```

`app/qa.py` is the single shared implementation used by both the `/ask` endpoint and this example — no logic duplicated between the API and scripts.

## Repository structure
```
.
├─ app/
│  ├─ config.py    # env-driven settings: models, chunk size, paths
│  ├─ ingest.py     # PDF -> page-aware overlapping chunks
│  ├─ embeddings.py # Ollama /api/embeddings wrapper
│  ├─ store.py      # ChromaDB add/query wrapper (lazy-initialized)
│  ├─ llm.py        # Ollama /api/generate wrapper, citation-constrained prompt
│  ├─ qa.py         # shared retrieve-then-generate logic (used by API + examples)
│  └─ main.py       # FastAPI app: /, /upload, /ask, /sources
├─ static/
│  └─ index.html    # single-file vanilla JS UI
├─ examples/
│  └─ quickstart.py # programmatic ingest + ask, no HTTP server
├─ experiments/
│  └─ matryoshka_benchmark.py  # embedding dimension truncation benchmark
├─ requirements.txt
├─ .env.example
└─ README.md
```

## Architecture

```
PDF upload
    │
    ▼
pypdf (per-page text extraction)
    │
    ▼
Overlapping chunking (chunk size 800, overlap 150)
    │
    ▼
Ollama embeddings (nomic-embed-text)
    │
    ▼
ChromaDB (local, on-disk, {source, page} metadata)

User question
    │
    ▼
Ollama embeddings (same model)
    │
    ▼
ChromaDB top-k similarity search
    │
    ▼
Ollama generation (qwen2.5:3b), context-constrained prompt
    │
    ▼
Answer + page-level citations
```

## Bonus experiment: Matryoshka embedding dimension truncation

`experiments/matryoshka_benchmark.py` tests whether embedding storage can be shrunk without hurting retrieval quality — the same experiment as the "Pocket Ranger" project ([Samhitha Muvva, on EmbeddingGemma](https://medium.com/@samhitha.muvva/embedding-gemma-promised-something-wild-turns-out-it-delivered-80180661e2f3)).

Google's `embeddinggemma` model is trained with Matryoshka Representation Learning (MRL): its 768-dimensional output is structured so the most important signal is packed into the first dimensions, meaning you can truncate it to 512/256/128 dims (with re-normalization) and keep most of its meaning — no re-embedding required.

**Run it:**
```bash
ollama pull embeddinggemma
python experiments/matryoshka_benchmark.py --pdf-dir data/uploads \
  --question "Your question about an uploaded PDF" \
  --question "Another question (repeatable)"
```

**Verified result** (8-page test PDF, real Ollama calls, not simulated): top-1 retrieval stayed identical across all four dimensions on every question, with overlap degrading gracefully (768→512: ~100%, →256: 67-100%, →128: 33-100%) while storage dropped to 1/6th. Matches the article's conclusion that 256 dimensions is often the sweet spot between quality and storage cost.

## Config reference

Set via environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `GEN_MODEL` | `qwen2.5:3b` | Generation model |
| `CHROMA_DIR` | `chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | `data/uploads` | Where uploaded PDFs are saved |

## Troubleshooting

- **`ConnectionError` / embeddings or generation calls fail** — make sure Ollama is running (`ollama serve` or the Ollama desktop app) and reachable at `OLLAMA_BASE_URL`.
- **`model not found` error from Ollama** — pull the missing model: `ollama pull <model-name>` (check `EMBED_MODEL`/`GEN_MODEL` in your `.env`).
- **`{"answer": "No documents have been indexed yet.", ...}`** — no PDFs uploaded yet, or `CHROMA_DIR` points at an empty/different directory than where you ingested.
- **Port already in use** — another `uvicorn` process is bound to 8000; stop it or run with `--port <other-port>`.

## Known limitations / next steps

- Retrieval is pure dense vector similarity — no hybrid BM25/keyword matching, so exact terms, codes, or numbers can be missed.
- No guardrail/eval layer — answers aren't automatically scored for groundedness (see DoorDash's LLM-judge pattern for a reference approach).
- Single-shot retrieval only — no multi-hop/agentic decomposition for questions that need combining facts across multiple documents.
- No multi-tenant access control — all uploaded documents are queryable by anyone hitting the API.
- Re-uploading a PDF re-embeds it from scratch; no incremental re-indexing.
