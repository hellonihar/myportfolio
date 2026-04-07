"""
FAISS vector store for storing and retrieving document chunk embeddings.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.config import settings


class VectorStore:
    """Manages a FAISS inner-product index with parallel metadata."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []  # parallel to FAISS rows
        self._index_dir = Path(settings.FAISS_INDEX_DIR)

    # ── Lifecycle ────────────────────────────────────────────

    def init_index(self, dimension: int) -> None:
        """Create a fresh FAISS index (or load from disk if available)."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_dir / "faiss.index"
        meta_path = self._index_dir / "metadata.json"

        if index_path.exists() and meta_path.exists():
            print("[VectorStore] Loading existing index from disk …")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"[VectorStore] Loaded {self.index.ntotal} vectors")
        else:
            print("[VectorStore] Creating new FAISS index …")
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = []

    def save(self) -> None:
        """Persist index + metadata to disk."""
        assert self.index is not None, "Index not initialised"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_dir / "faiss.index"))
        with open(self._index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"[VectorStore] Saved {self.index.ntotal} vectors to disk")

    # ── Operations ───────────────────────────────────────────

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        """Add embeddings and their metadata to the store."""
        assert self.index is not None, "Index not initialised"
        assert len(embeddings) == len(metadatas)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
        """
        Search for the most similar vectors.

        Returns a list of dicts: { text, source, chunk_index, score }
        """
        assert self.index is not None, "Index not initialised"
        k = min(top_k or settings.TOP_K, self.index.ntotal)
        if k == 0:
            return []

        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = {**self.metadata[idx], "score": float(score)}
            results.append(entry)
        return results

    @property
    def total_chunks(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal


# Singleton
vector_store = VectorStore()
