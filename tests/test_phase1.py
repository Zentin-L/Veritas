from veritas.chunking import chunk_pages
from veritas.ingest import clean_text
from veritas.schemas import PageText


def test_clean_text_removes_hyphen_breaks_and_repeated_lines() -> None:
    text = "Header\ninter-\nnational research.\nFooter"
    assert clean_text(text, {"Header", "Footer"}) == "international research."


def test_chunking_preserves_metadata_and_has_no_empty_chunks() -> None:
    page = PageText("guide.pdf", 12, "One short sentence. Another useful sentence. Final sentence.")
    chunks = chunk_pages([page], max_tokens=4, overlap_tokens=1)
    assert chunks
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.doc_name == "guide.pdf" and chunk.page == 12 for chunk in chunks)
    assert chunks[0].id == "guide.pdf-p12-c1"
    assert chunks[0].start_char == 0


def test_chunking_overlaps_whole_sentences() -> None:
    page = PageText("guide.pdf", 1, "First sentence. Second sentence. Third sentence.")
    chunks = chunk_pages([page], max_tokens=4, overlap_tokens=2)
    assert len(chunks) == 2
    assert "Second sentence." in chunks[0].text
    assert "Second sentence." in chunks[1].text
