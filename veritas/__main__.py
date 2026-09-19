"""Command-line entry points for Veritas."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .embeddings import LocalEmbedder
from .generate import answer_question
from .index import SearchIndex
from .ingest import extract_pdf
from .llm import configured_provider
from .retrieve import retrieve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veritas", description="Zero-cost, source-grounded PDF knowledge assistant")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Ingest PDFs into a local searchable index")
    ingest.add_argument("pdfs", nargs="+", help="One or more PDF files to ingest")
    ingest.add_argument("--output", default="storage/index", help="Directory to store the generated index")

    ask = subparsers.add_parser("ask", help="Ask a question against the local index")
    ask.add_argument("question", help="The question to answer")
    ask.add_argument("--index", default="storage/index", help="Directory containing the built index")

    return parser


def _source_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _run_ingest(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    settings = Settings.from_env()
    pages: list = []
    for pdf_path in args.pdfs:
        pages.extend(extract_pdf(pdf_path))
    if not pages:
        print("No text was found in the provided PDFs.")
        return 1
    chunks = chunk_pages(pages)
    vectors = LocalEmbedder(settings).embed_passages([chunk.text for chunk in chunks])
    index = SearchIndex.build(chunks, vectors, _source_hash(args.pdfs))
    index.save(output_dir)
    print(f"Indexed {len(chunks)} chunks from {len(args.pdfs)} PDF(s) into {output_dir}")
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    index_dir = Path(args.index)
    if not index_dir.exists():
        print(f"Index not found: {index_dir}")
        return 1
    settings = Settings.from_env()
    index = SearchIndex.load(index_dir)
    embedder = LocalEmbedder(settings)
    results = retrieve(index, args.question, embedder.embed_query, settings)
    provider = configured_provider(settings)
    answer = answer_question(args.question, results, provider, settings)
    if answer.abstain:
        print("No answer could be supported by the local source documents.")
        return 0
    joined = " ".join(sentence.text for sentence in answer.sentences)
    print(joined)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "ask":
        return _run_ask(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
