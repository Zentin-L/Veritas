"""Hybrid dense/sparse retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from .config import Settings
from .index import SearchIndex
from .schemas import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    """Expose component scores so the UI can explain why a chunk was selected."""

    chunk: Chunk
    dense_score: float
    bm25_score: float
    rrf_score: float


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def reciprocal_rank_fusion(
    dense_ids: list[int], sparse_ids: list[int], k: int = 60
) -> list[tuple[int, float]]:
    """Combine rankings without pretending dense and BM25 scores share a scale."""
    scores: dict[int, float] = {}
    for rank, index in enumerate(dense_ids):
        scores[index] = scores.get(index, 0.0) + 1 / (k + rank + 1)
    for rank, index in enumerate(sparse_ids):
        scores[index] = scores.get(index, 0.0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    """Load the optional reranker lazily because it is unnecessary by default."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def retrieve(
    index: SearchIndex,
    query: str,
    embed_query: Callable[[str], np.ndarray],
    settings: Settings | None = None,
) -> list[RetrievalResult]:
    """Return top hybrid matches, optionally reordered by a local cross-encoder."""
    settings = settings or Settings()
    query_vector = np.asarray(embed_query(query), dtype="float32").reshape(1, -1)
    dense_scores, dense_indices = index.dense.search(query_vector, settings.candidate_k)
    sparse_scores = np.asarray(index.bm25.get_scores(_tokens(query)))
    sparse_indices = np.argsort(-sparse_scores)[: settings.candidate_k].tolist()
    dense_ids = [int(item) for item in dense_indices[0] if item >= 0]
    dense_score_by_id = {
        int(item): float(score)
        for item, score in zip(dense_indices[0], dense_scores[0])
        if item >= 0
    }
    fused = reciprocal_rank_fusion(dense_ids, sparse_indices, settings.rrf_k)
    candidates = [
        RetrievalResult(index.chunks[item], dense_score_by_id.get(item, 0.0), float(sparse_scores[item]), score)
        for item, score in fused
    ]
    if settings.rerank and candidates:
        reranker = _load_reranker(settings.reranker_model)
        scores = reranker.predict([(query, result.chunk.text) for result in candidates])
        candidates = [result for _, result in sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)]
    return candidates[: settings.top_k]
