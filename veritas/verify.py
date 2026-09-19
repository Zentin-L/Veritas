"""Verify generated claims against local source text with NLI."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Protocol, Sequence

from .config import Settings
from .retrieve import RetrievalResult
from .schemas import ClaimVerification, GeneratedAnswer, VerificationResult

_SENTENCE = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.DOTALL)


class ClaimVerifier(Protocol):
    """Small interface allowing deterministic tests and a local production model."""

    def logits(self, premise: str, hypothesis: str) -> Sequence[float]:
        ...


def _softmax(logits: Sequence[float]) -> list[float]:
    if len(logits) != 3:
        raise ValueError("NLI classifier must return three logits")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _windows(text: str, size: int = 3, overlap: int = 1) -> list[str]:
    """Use overlapping sentence windows so evidence is not lost at chunk boundaries."""
    sentences = [match.group().strip() for match in _SENTENCE.finditer(text) if match.group().strip()]
    if not sentences:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(sentences[start : start + size]) for start in range(0, len(sentences), step)]


@lru_cache(maxsize=1)
def _load_nli(model_name: str):
    """Load the NLI model only when verification is first requested."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu", num_labels=3)
    labels = getattr(getattr(model, "model", None), "config", None)
    label_map = getattr(labels, "id2label", {}) if labels else {}
    if len(label_map) not in (0, 3):
        raise ValueError("NLI classifier must have exactly three labels")
    return model


class NLIClaimVerifier:
    """Run local NLI logits using the required contradiction/entailment/neutral order."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def logits(self, premise: str, hypothesis: str) -> Sequence[float]:
        model = _load_nli(self.model_name)
        values = model.predict([(premise, hypothesis)], apply_softmax=False)
        return values[0]


def _claim_score(verifier: ClaimVerifier, claim: str, passages: Sequence[str]) -> float:
    scores = []
    for passage in passages:
        for window in _windows(passage):
            probabilities = _softmax(verifier.logits(window, claim))
            scores.append(probabilities[1])
    return max(scores, default=0.0)


def verify_answer(
    answer: GeneratedAnswer,
    results: Sequence[RetrievalResult],
    verifier: ClaimVerifier,
    settings: Settings | None = None,
) -> VerificationResult:
    """Score every cited claim and abstain when aggregate trust is too low."""
    settings = settings or Settings()
    by_id = {result.chunk.id: result.chunk.text for result in results}
    claims: list[ClaimVerification] = []
    for sentence in answer.sentences:
        passages = [by_id[citation] for citation in sentence.citations if citation in by_id]
        score = _claim_score(verifier, sentence.text, passages)
        label = "supported" if score >= settings.supported_threshold else "partial" if score >= settings.partial_threshold else "unsupported"
        claims.append(ClaimVerification(text=sentence.text, citations=sentence.citations, entailment_probability=score, label=label))
    total = len(claims)
    supported = sum(claim.label == "supported" for claim in claims)
    partial = sum(claim.label == "partial" for claim in claims)
    trust = (supported + 0.5 * partial) / total if total else 0.0
    abstain = answer.abstain or trust < settings.trust_threshold
    passages = [result.chunk.text for result in results[: settings.top_k]] if abstain else []
    return VerificationResult(abstain=abstain, trust_score=trust, claims=claims, source_passages=passages)