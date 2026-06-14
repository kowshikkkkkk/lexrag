import pytest
from ingestion.loader import Document
from chunking.splitter import split_document, recursive_split


def test_recursive_split_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = recursive_split(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) > 0
    assert all(len(c) <= 60 for c in chunks)


def test_recursive_split_preserves_content():
    text = "Hello world. Goodbye world."
    chunks = recursive_split(text, chunk_size=100, chunk_overlap=0)
    combined = " ".join(chunks)
    assert "Hello" in combined
    assert "Goodbye" in combined


def test_split_document_returns_chunks():
    doc = Document(
        text="Section 378\nTheft definition here.\n\nSection 379\nPunishment here.",
        metadata={"source": "test.txt", "doc_type": "act", "file_hash": "abc123"},
    )
    chunks = split_document(doc)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


def test_split_document_chunk_metadata():
    doc = Document(
        text="Some legal text here for testing purposes.",
        metadata={"source": "test.txt", "doc_type": "generic", "file_hash": "abc123"},
    )
    chunks = split_document(doc)
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == len(chunks)
    assert chunks[0].chunk_id is not None


def test_split_document_empty_raises():
    from config.exceptions import ChunkingError
    doc = Document(text="", metadata={"source": "test.txt", "doc_type": "generic", "file_hash": "abc123"})
    with pytest.raises(ChunkingError):
        split_document(doc)


def test_section_aware_split():
    text = """Section 378
Theft definition.

Section 379
Punishment for theft."""
    doc = Document(
        text=text,
        metadata={"source": "test.txt", "doc_type": "act", "file_hash": "abc123"},
    )
    chunks = split_document(doc)
    sections = [c.metadata.get("section_number") for c in chunks]
    assert "Section 378" in sections
    assert "Section 379" in sections