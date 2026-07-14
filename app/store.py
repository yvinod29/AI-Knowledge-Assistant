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
    metadatas = [{k: v for k, v in chunk.items() if k != "text"} for chunk in chunks]
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


def delete_source(source: str) -> None:
    _get_collection().delete(where={"source": source})


def get_source_content_hash(source: str) -> str | None:
    data = _get_collection().get(where={"source": source}, limit=1)
    if not data["metadatas"]:
        return None
    return data["metadatas"][0].get("content_hash")


def list_sources() -> list[dict]:
    data = _get_collection().get()
    sources: dict[str, dict] = {}
    for metadata in data["metadatas"]:
        source = metadata["source"]
        if source not in sources:
            sources[source] = {
                "source": source,
                "content_hash": metadata.get("content_hash"),
                "indexed_at": metadata.get("indexed_at"),
            }
    return sorted(sources.values(), key=lambda s: s["source"])
