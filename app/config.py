import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:3b")

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "documents"

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data/uploads")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
