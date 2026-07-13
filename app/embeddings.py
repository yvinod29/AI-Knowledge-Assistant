import requests

from app.config import OLLAMA_BASE_URL, EMBED_MODEL


def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    return [embed_text(text, model=model) for text in texts]
