import hashlib
from datetime import datetime, timezone

from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.embeddings import embed_batch
from app.store import add_chunks, delete_source, get_source_content_hash


def chunk_page_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def extract_chunks(pdf_path: str, source_name: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in chunk_page_text(text):
            chunks.append({"text": piece, "source": source_name, "page": page_number})
    return chunks


def file_content_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ingest_pdf(pdf_path: str, source_name: str) -> dict:
    """Ingest a PDF, invalidating any stale index for the same source.

    A document is re-indexed only if its content actually changed (by
    content hash), so an unmodified re-upload is a cheap no-op, and an
    updated document has its old chunks removed before the new ones are
    embedded — otherwise stale chunks from the old version would keep
    getting retrieved alongside (or instead of) the current ones.
    """
    new_hash = file_content_hash(pdf_path)
    existing_hash = get_source_content_hash(source_name)

    if existing_hash == new_hash:
        return {"status": "unchanged", "chunks_indexed": 0}

    chunks = extract_chunks(pdf_path, source_name)
    if not chunks:
        return {"status": "empty", "chunks_indexed": 0}

    if existing_hash is not None:
        delete_source(source_name)

    indexed_at = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk["content_hash"] = new_hash
        chunk["indexed_at"] = indexed_at

    embeddings = embed_batch([chunk["text"] for chunk in chunks])
    add_chunks(chunks, embeddings)

    status = "updated" if existing_hash is not None else "created"
    return {"status": status, "chunks_indexed": len(chunks)}
