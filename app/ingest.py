from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.embeddings import embed_batch
from app.store import add_chunks


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


def ingest_pdf(pdf_path: str, source_name: str) -> int:
    chunks = extract_chunks(pdf_path, source_name)
    if not chunks:
        return 0

    embeddings = embed_batch([chunk["text"] for chunk in chunks])
    add_chunks(chunks, embeddings)
    return len(chunks)
