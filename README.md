# RAG

A Retrieval-Augmented Generation system with semantic search, a streaming web UI, and built-in evaluation. Ask natural-language questions and get answers grounded in your own documents.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Docker](https://img.shields.io/badge/docker-ready-blue)

## How it works

```
Question → Embed (MiniLM-L6) → FAISS similarity search → Top chunks → Llama 3.3 70B via Groq → Streamed answer
```

Documents are split with sentence-aware chunking (respecting paragraph and sentence boundaries), embedded into 384-dimensional vectors using `all-MiniLM-L6-v2`, and indexed in a FAISS vector store. At query time, the question is embedded the same way, the most similar chunks are retrieved, and sent to Llama 3.3 70B through Groq's API, which streams an answer grounded only in the retrieved context. Conversation history is maintained per session for multi-turn Q&A.

## Quick start

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your-key-here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

```bash
python app.py    # opens web UI at http://localhost:5000
```

Or use the CLI:

```bash
python rag.py                      # interactive mode
python rag.py "your question"      # single question
```

## Adding documents

**Via the web UI** — click "Add documents" or drag and drop files directly onto the page. Supports `.txt`, `.md`, `.json`, `.csv`, `.pdf`, and `.docx`.

**Via JSON** — edit `data/documents.json`. Each entry needs `id`, `title`, and `content`:

```json
[
  {
    "id": "onboarding",
    "title": "Onboarding Guide",
    "content": "New hires should complete orientation within their first week..."
  }
]
```

**Swapping the dataset** — replace `data/documents.json` with any JSON array of `{id, title, content}` objects. The system works with any domain — company policies, product docs, research papers, game guides, etc.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check (for deployment) |
| `GET` | `/stats` | Index statistics (chunk count, doc count, model info) |
| `POST` | `/ask` | Stream an answer (JSON: `{query, session_id}`) |
| `POST` | `/upload` | Upload documents (multipart form) |
| `POST` | `/evaluate` | Run retrieval evaluation (JSON or uses `data/eval.json`) |
| `POST` | `/clear` | Clear conversation history (JSON: `{session_id}`) |

## Evaluation

Run retrieval quality tests:

```bash
curl -X POST http://localhost:5000/evaluate | python -m json.tool
```

This uses the test questions in `data/eval.json`. You can also POST custom questions:

```json
{
  "questions": [
    {"query": "What is X?", "expected_sources": ["DocumentTitle"]}
  ]
}
```

Returns accuracy, per-question hits, and retrieved sources.

## Docker

```bash
docker build -t rag .
docker run -p 5000:5000 -e GROQ_API_KEY=your-key rag
```

## Deploy to Railway

1. Push to GitHub
2. Connect repo to [Railway](https://railway.app)
3. Add `GROQ_API_KEY` as an environment variable
4. Deploy — Railway auto-detects the Dockerfile

## Tests

```bash
python tests.py
```

Covers sentence splitting, chunking, vector search semantics, and index operations.

## Project structure

```
├── app.py              # Flask web UI + API (streaming, uploads, eval, history)
├── rag.py              # Core pipeline (load, chunk, embed, search, generate)
├── tests.py            # Unit tests
├── Dockerfile
├── requirements.txt
├── .gitignore
└── data/
    ├── documents.json  # Source documents (swap this for any domain)
    ├── eval.json       # Evaluation questions
    └── uploads/        # User-uploaded files (gitignored)
```

## Architecture

| Component | Implementation |
|-----------|---------------|
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (384-dim) |
| Vector store | FAISS `IndexFlatIP` (exact inner product search) |
| LLM | Llama 3.3 70B via Groq API (streaming) |
| Chunking | Sentence-aware with configurable overlap |
| File parsing | PDF (pdfplumber), DOCX (python-docx), TXT, JSON, CSV, MD |
| Serving | Flask with SSE streaming |
| Conversation | In-memory session history (last 10 turns) |
| Evaluation | Retrieval accuracy scoring via `/evaluate` |
| Deployment | Docker + Railway/Render ready |

## Notes on how this was built

This project was built with Claude as a coding partner. I designed the requirements, architecture, and worked through the code to understand how each piece works. I'm happy to explain decisions or details further.