"""Persistent FAISS and BM25 indexes over the same chunk list."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import Chunk


@dataclass
class SearchIndex:
    """Keep dense and sparse indexes aligned by their shared chunk positions."""

    chunks: list[Chunk]
    dense: faiss.Index
    bm25: BM25Okapi
    source_hash: str

    @classmethod
    def build(
        cls, chunks: Sequence[Chunk], vectors: np.ndarray, source_hash: str
    ) -> "SearchIndex":
        if len(chunks) == 0 or len(vectors) != len(chunks):
            raise ValueError("chunks and vectors must be non-empty and have equal length")
        vectors = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        dense = faiss.IndexFlatIP(vectors.shape[1])
        dense.add(vectors)
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        return cls(list(chunks), dense, BM25Okapi(tokenized), source_hash)

    def save(self, directory: str | Path) -> None:
        """Persist all aligned state so startup does not recompute local indexes."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.dense, str(target / "dense.faiss"))
        with (target / "bm25.pkl").open("wb") as file:
            pickle.dump(self.bm25, file)
        metadata = {"source_hash": self.source_hash, "chunks": [chunk.__dict__ for chunk in self.chunks]}
        (target / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def load_or_build(
        cls,
        directory: str | Path,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        source_hash: str,
    ) -> "SearchIndex":
        """Reuse persisted state when the document version is unchanged."""
        target = Path(directory)
        metadata_path = target / "metadata.json"
        if metadata_path.exists():
            try:
                return cls.load(target, source_hash)
            except (OSError, ValueError, pickle.PickleError, json.JSONDecodeError):
                pass
        index = cls.build(chunks, vectors, source_hash)
        index.save(target)
        return index

    @classmethod
    def load(cls, directory: str | Path, source_hash: str | None = None) -> "SearchIndex":
        """Reload only a matching document version, preventing stale citations."""
        target = Path(directory)
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if source_hash is not None and metadata["source_hash"] != source_hash:
            raise ValueError("stored index does not match the current document hash")
        chunks = [Chunk(**item) for item in metadata["chunks"]]
        with (target / "bm25.pkl").open("rb") as file:
            bm25 = pickle.load(file)
        return cls(chunks, faiss.read_index(str(target / "dense.faiss")), bm25, metadata["source_hash"])
