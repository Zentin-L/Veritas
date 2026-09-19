"""Sentence-aware chunking for source-grounded retrieval."""

from __future__ import annotations

import re

from .schemas import Chunk, PageText

_SENTENCE = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.DOTALL)
_TOKEN = re.compile(r"\S+")


def _sentences(text: str) -> list[tuple[str, int, int]]:
    """Return sentence spans so chunks never split a sentence in the middle."""
    result: list[tuple[str, int, int]] = []
    for match in _SENTENCE.finditer(text):
        sentence = match.group().strip()
        if sentence:
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            result.append((sentence, start, start + len(sentence)))
    return result


def chunk_pages(
    pages: list[PageText], max_tokens: int = 400, overlap_tokens: int = 60
) -> list[Chunk]:
    """Build retrievable passages with sentence overlap and page-relative offsets."""
    if max_tokens <= 0 or overlap_tokens < 0:
        raise ValueError("max_tokens must be positive and overlap_tokens cannot be negative")

    chunks: list[Chunk] = []
    for page in pages:
        sentences = _sentences(page.text)
        start_index = 0
        chunk_number = 1
        while start_index < len(sentences):
            end_index = start_index
            token_count = 0
            while end_index < len(sentences):
                sentence_tokens = len(_TOKEN.findall(sentences[end_index][0]))
                if end_index > start_index and token_count + sentence_tokens > max_tokens:
                    break
                token_count += sentence_tokens
                end_index += 1
            selected = sentences[start_index:end_index]
            text = " ".join(sentence[0] for sentence in selected)
            chunks.append(
                Chunk(
                    id=f"{page.doc_name}-p{page.page_number}-c{chunk_number}",
                    text=text,
                    doc_name=page.doc_name,
                    page=page.page_number,
                    start_char=selected[0][1],
                    end_char=selected[-1][2],
                )
            )
            if end_index == len(sentences):
                break
            overlap = 0
            next_start = end_index
            while next_start > start_index and overlap < overlap_tokens:
                next_start -= 1
                overlap += len(_TOKEN.findall(sentences[next_start][0]))
            start_index = max(next_start, start_index + 1)
            chunk_number += 1
    return chunks
