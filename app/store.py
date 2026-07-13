import uuid
from functools import lru_cache

import chromadb

from app.config import CHROMA_DIR, COLLECTION_NAME


@lru_cache
def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "page": chunk["page"]} for chunk in chunks
    ]
    _get_collection().add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )


def query(embedding: list[float], top_k: int) -> list[dict]:
    result = _get_collection().query(query_embeddings=[embedding], n_results=top_k)
    chunks = []
    for text, metadata in zip(result["documents"][0], result["metadatas"][0]):
        chunks.append(
            {"text": text, "source": metadata["source"], "page": metadata["page"]}
        )
    return chunks


def list_sources() -> list[str]:
    data = _get_collection().get()
    sources = {metadata["source"] for metadata in data["metadatas"]}
    return sorted(sources)
