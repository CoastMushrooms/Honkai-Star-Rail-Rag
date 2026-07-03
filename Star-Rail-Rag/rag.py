"""
Core RAG pipeline: load, chunk, embed, search, generate.
Supports JSON, PDF, DOCX, TXT, MD, CSV ingestion.
Uses sentence-aware chunking and FAISS + sentence-transformers.
"""

import json
import os
import re
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Document flattening (nested dicts → readable text)
# ---------------------------------------------------------------------------

def flatten(obj, prefix=""):
    parts = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            label = key.replace("_", " ").title()
            if isinstance(value, dict):
                parts.append(f"\n{label}:")
                parts.append(flatten(value, prefix=f"  "))
            else:
                parts.append(f"{prefix}{label}: {value}")
    elif isinstance(obj, str):
        parts.append(f"{prefix}{obj}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# File parsers (PDF, DOCX, plain text)
# ---------------------------------------------------------------------------

def parse_pdf(file_bytes):
    """Extract text from PDF bytes using pdfplumber (falls back to PyPDF2)."""
    import io
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        raise ImportError("Install pdfplumber or PyPDF2 to read PDFs: pip install pdfplumber")


def parse_docx(file_bytes):
    """Extract text from DOCX bytes."""
    import io
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx to read DOCX files: pip install python-docx")
    doc = Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_docs(path="data/documents.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = []
    for doc in raw:
        content = doc["content"]
        text = flatten(content) if isinstance(content, dict) else content
        documents.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": text,
        })
    return documents


# ---------------------------------------------------------------------------
# Sentence-aware chunking
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')


def _split_sentences(text):
    """Split text into sentences/paragraphs while preserving boundaries."""
    parts = _SENT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def chunk_docs(documents, max_chars=500, overlap_sentences=1):
    """
    Sentence-aware chunking:
    1. Split each document into sentences/paragraphs.
    2. Greedily group sentences until max_chars is reached.
    3. Overlap by re-including the last N sentences from the previous chunk.
    """
    chunks = []
    for doc in documents:
        text = doc["content"]
        title = doc["title"]
        doc_id = doc["id"]

        sentences = _split_sentences(text)
        if not sentences:
            continue

        # If entire doc fits in one chunk, just use it
        if len(text) <= max_chars:
            chunks.append({
                "id": f"{doc_id}-0",
                "text": f"{title}\n{text}",
                "title": title,
                "source_id": doc_id,
            })
            continue

        current = []
        current_len = 0
        chunk_num = 0

        for sent in sentences:
            # If adding this sentence exceeds limit and we have content, flush
            if current and current_len + len(sent) + 1 > max_chars:
                chunks.append({
                    "id": f"{doc_id}-{chunk_num}",
                    "text": f"{title}\n" + " ".join(current),
                    "title": title,
                    "source_id": doc_id,
                })
                chunk_num += 1
                # Overlap: keep last N sentences
                overlap = current[-overlap_sentences:] if overlap_sentences else []
                current = list(overlap)
                current_len = sum(len(s) for s in current) + len(current)

            current.append(sent)
            current_len += len(sent) + 1

        # Flush remaining
        if current:
            chunks.append({
                "id": f"{doc_id}-{chunk_num}",
                "text": f"{title}\n" + " ".join(current),
                "title": title,
                "source_id": doc_id,
            })

    return chunks


# ---------------------------------------------------------------------------
# Vector store (FAISS)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, chunks=None):
        print(f"Loading embedding model ({EMBED_MODEL})...")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"Indexed {len(self.chunks)} chunks ({self.dim}-dim embeddings)")

    def search(self, query, n_results=5):
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, min(n_results, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                **self.chunks[idx],
                "score": float(score),
            })
        return results

    def stats(self):
        """Return index statistics."""
        titles = set(c["title"] for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(titles),
            "embedding_dim": self.dim,
            "model": EMBED_MODEL,
        }


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(query, retrieved_chunks):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer based only on the provided context. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("  Set your GROQ_API_KEY first:")
        print("  $env:GROQ_API_KEY='your-key-here'")
        print("  Get one free at: https://console.groq.com/keys")
        sys.exit(1)

    print("Loading documents...")
    docs = load_docs()
    chunks = chunk_docs(docs)
    store = VectorStore(chunks)
    print()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Q: {query}\n")
        results = store.search(query)
        print(f"Retrieved {len(results)} chunks:")
        for r in results:
            print(f"   - {r['title']} (score: {r['score']:.3f})")
        print()
        answer = generate_answer(query, results)
        print(f"A: {answer}")
        return

    print("Ask questions about your documents! Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = store.search(query)
        answer = generate_answer(query, results)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
