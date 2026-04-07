"""
Application configuration using Pydantic Settings.
All values can be overridden via environment variables or a .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

# Project root is one level above the app/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the RAG system."""

    # ── Embedding ────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Ollama / LLM ────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    # ── Chunking ─────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Retrieval ────────────────────────────────────────────
    TOP_K: int = 5

    # ── Paths ────────────────────────────────────────────────
    FAISS_INDEX_DIR: str = str(PROJECT_ROOT / "data" / "index")
    UPLOAD_DIR: str = str(PROJECT_ROOT / "data" / "uploads")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
    }


settings = Settings()
