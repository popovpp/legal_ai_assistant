# tests/unit/test_chunker.py
from app.ingestion.chunker import chunk_pages
from app.ingestion.extractor import PageText


def test_chunker_splits_by_page():
    pages = [
        PageText(page=1, text="A" * 1500),
        PageText(page=2, text="B" * 1500),
    ]
    chunks = chunk_pages("doc1", pages, chunk_size=500, chunk_overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert c.chunk_id.startswith("doc1:")
        assert c.char_end > c.char_start


def test_chunker_skips_empty_pages():
    pages = [PageText(page=1, text="   "), PageText(page=2, text="real text")]
    chunks = chunk_pages("doc1", pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_chunker_ids_are_unique():
    pages = [PageText(page=1, text="x " * 5000)]
    chunks = chunk_pages("doc1", pages, chunk_size=300, chunk_overlap=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
