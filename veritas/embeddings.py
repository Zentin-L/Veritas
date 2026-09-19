"""Local passage/query embeddings with a small disk cache."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import Settings


def _cache_key(texts: Sequence[str], model_name: str) -> str:
    payload = json.dumps([model_name, *texts], ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load the encoder only when an embedding is actually requested."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


class LocalEmbedder:
    """Wrap bge so query instructions are never accidentally used for passages."""

    def __init__(self, settings: Settings | None = None, cache_dir: str | Path = "storage/embeddings"):
        self.settings = settings or Settings()
        self.cache_dir = Path(cache_dir)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embed source text without the bge query instruction prefix."""
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query using bge's required retrieval instruction."""
        return self._encode([self.settings.query_prefix + query])[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{_cache_key(texts, self.settings.embedding_model)}.npy"
        if path.exists():
            return np.load(path)
        vectors = _load_model(self.settings.embedding_model).encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        vectors = np.asarray(vectors, dtype="float32")
        np.save(path, vectors)
        return vectors
