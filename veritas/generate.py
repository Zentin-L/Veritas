"""Create short, citation-complete answers from retrieved source chunks."""

from __future__ import annotations

import json
import re
from typing import Sequence

from .config import Settings
from .llm import LLMProvider
from .retrieve import RetrievalResult
from .schemas import GeneratedAnswer

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json(raw: str) -> GeneratedAnswer:
    """Parse the first JSON object while tolerating fences and trailing commentary."""
    cleaned = _CODE_FENCE.sub("", raw).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as error:
        raise ValueError("model response is not valid JSON") from error
    return GeneratedAnswer.model_validate(value)


def _context(results: Sequence[RetrievalResult], budget: int) -> tuple[str, set[str]]:
    """Keep complete highest-ranked chunks until the prompt budget is reached."""
    parts: list[str] = []
    ids: set[str] = set()
    used = 0
    for result in results:
        tokens = result.chunk.text.split()
        if used + len(tokens) > budget:
            continue
        parts.append(f"[{result.chunk.id}] {result.chunk.text}")
        ids.add(result.chunk.id)
        used += len(tokens)
    return "\n\n".join(parts), ids


def _prompt(query: str, context: str) -> str:
    return f'''Answer only from the context. Every sentence must cite at least one context ID.
If the context is insufficient, set "abstain" to true and use an empty sentences list.
Return JSON only with this shape: {{"abstain": false, "sentences": [{{"text": "...", "citations": ["chunk-id"]}}]}}
Example: {{"abstain": false, "sentences": [{{"text": "The sky is blue.", "citations": ["doc-p1-c1"]}}]}}

Question: {query}
Context:
{context}'''


def _validated_answer(raw: str, valid_ids: set[str]) -> GeneratedAnswer:
    answer = extract_json(raw)
    sentences = []
    for sentence in answer.sentences:
        citations = [citation for citation in sentence.citations if citation in valid_ids]
        if citations:
            sentences.append(sentence.model_copy(update={"citations": citations}))
    if answer.sentences and not sentences:
        return GeneratedAnswer(abstain=True, sentences=[])
    return answer.model_copy(update={"sentences": sentences})


def answer_question(
    query: str,
    results: Sequence[RetrievalResult],
    provider: LLMProvider,
    settings: Settings | None = None,
) -> GeneratedAnswer:
    """Gate low-confidence retrieval and make at most two generation attempts."""
    settings = settings or Settings()
    if not results or max(result.dense_score for result in results) < settings.confidence_threshold:
        return GeneratedAnswer(abstain=True, sentences=[])
    context, valid_ids = _context(results, settings.context_token_budget)
    if not context:
        return GeneratedAnswer(abstain=True, sentences=[])
    prompt = _prompt(query, context)
    try:
        return _validated_answer(provider.generate(prompt), valid_ids)
    except (ValueError, TypeError):
        retry = prompt + "\nReturn valid JSON only. Do not add markdown or commentary."
        try:
            return _validated_answer(provider.generate(retry), valid_ids)
        except (ValueError, TypeError, RuntimeError):
            return GeneratedAnswer(abstain=True, sentences=[])
    except RuntimeError:
        return GeneratedAnswer(abstain=True, sentences=[])
