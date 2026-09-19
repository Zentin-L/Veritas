"""Extract and normalize PDF text while preserving page provenance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import fitz

from .schemas import PageText


_WHITESPACE = re.compile(r"\s+")
_HYPHEN_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def clean_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """Normalize layout artifacts so chunk boundaries reflect sentences, not PDF lines."""
    repeated_lines = repeated_lines or set()
    kept_lines: list[str] = []
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if normalized and normalized not in repeated_lines:
            kept_lines.append(normalized)
    cleaned = "\n".join(kept_lines)
    cleaned = _HYPHEN_BREAK.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _repeated_lines(raw_pages: Iterable[str]) -> set[str]:
    pages = list(raw_pages)
    if len(pages) < 2:
        return set()
    counts: dict[str, int] = {}
    for raw_page in pages:
        lines = {" ".join(line.split()) for line in raw_page.splitlines() if line.strip()}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
    limit = len(pages) / 2
    return {line for line, count in counts.items() if count > limit}


def extract_pdf(path: str | Path) -> list[PageText]:
    """Extract pages first, then remove repeated headers and footers consistently."""
    pdf_path = Path(path)
    with fitz.open(pdf_path) as document:
        raw_pages = [page.get_text("text") for page in document]
    if not any(page.strip() for page in raw_pages):
        raise ValueError(f"PDF contains no extractable text: {pdf_path.name}")
    repeated = _repeated_lines(raw_pages)
    pages = [
        PageText(pdf_path.name, number, clean_text(raw, repeated))
        for number, raw in enumerate(raw_pages, start=1)
    ]
    return [page for page in pages if page.text]
