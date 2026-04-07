"""
SentenceTransformer wrapper for generating text embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    """Thin wrapper around SentenceTransformer for encoding text to vectors."""

    def __init__(self) -> None:
        self.model_name = settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """Load the model (called once at app startup)."""
        print(f"[Embedder] Loading model: {self.model_name} ...")
        self._model = SentenceTransformer(self.model_name)
        print(f"[Embedder] Model loaded — dimension: {self.dimension}")

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        assert self._model is not None, "Embedder not loaded"
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised embeddings.

        Returns:
            np.ndarray of shape (len(texts), dimension) with dtype float32.
        """
        assert self._model is not None, "Embedder not loaded"
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-length for cosine via inner-product
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


# Singleton – loaded in lifespan
embedder = Embedder()
