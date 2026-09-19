from pathlib import Path

import fitz
import numpy as np

from veritas.__main__ import main
from veritas.index import SearchIndex
from veritas.schemas import Chunk


def test_main_help_lists_subcommands(capsys) -> None:
    exit_code = main(["--help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    text = captured.out.lower()
    assert "ingest" in text
    assert "ask" in text


def test_ingest_creates_index_from_pdf(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((20, 20), "This is a test sentence. Another sentence follows.")
    document.save(pdf_path)
    document.close()

    output_dir = tmp_path / "index"
    result = main(["ingest", str(pdf_path), "--output", str(output_dir)])

    assert result == 0
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "dense.faiss").exists()


def test_ask_loads_index_and_returns_zero(monkeypatch, tmp_path) -> None:
    index_dir = tmp_path / "index"
    chunk = Chunk("doc-p1-c1", "Cats purr softly.", "doc", 1, 0, 11)
    built = SearchIndex.build([chunk], np.eye(1, dtype="float32"), "hash")
    built.save(index_dir)

    import veritas.__main__ as cli

    monkeypatch.setattr(cli, "LocalEmbedder", lambda *args, **kwargs: type("E", (), {"embed_query": lambda self, query: np.array([1.0], dtype="float32")})())
    monkeypatch.setattr(cli, "retrieve", lambda *args, **kwargs: [type("R", (), {"chunk": chunk, "dense_score": 1.0, "bm25_score": 0.5, "rrf_score": 0.8})()])
    monkeypatch.setattr(cli, "configured_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "answer_question", lambda *args, **kwargs: type("A", (), {"abstain": False, "sentences": [type("S", (), {"text": "Cats purr softly."})()]})())

    result = main(["ask", "What do cats do?", "--index", str(index_dir)])
    assert result == 0
