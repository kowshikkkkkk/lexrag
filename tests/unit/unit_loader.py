import pytest
from pathlib import Path
from ingestion.loader import load_document, normalize_text, compute_file_hash
from config.exceptions import UnsupportedFileTypeError, IngestionError


def test_normalize_text_removes_extra_whitespace():
    text = "Hello   world\n\n\n\nGoodbye"
    result = normalize_text(text)
    assert "   " not in result
    assert result.count("\n\n\n") == 0


def test_normalize_text_unicode():
    text = "Section\u00a0420"  # non-breaking space
    result = normalize_text(text)
    assert result == "Section 420"


def test_load_txt_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("This is a test legal document.")
    doc = load_document(str(f), "act")
    assert doc.text == "This is a test legal document."
    assert doc.metadata["source"] == "test.txt"
    assert doc.metadata["doc_type"] == "act"
    assert doc.metadata["file_hash"] is not None


def test_unsupported_file_type(tmp_path):
    f = tmp_path / "test.xlsx"
    f.write_text("content")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(str(f), "act")


def test_file_not_found():
    with pytest.raises(IngestionError):
        load_document("nonexistent.txt", "act")


def test_file_hash_consistency(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Same content")
    hash1 = compute_file_hash(f)
    hash2 = compute_file_hash(f)
    assert hash1 == hash2


def test_different_files_different_hash(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("Content A")
    f2.write_text("Content B")
    assert compute_file_hash(f1) != compute_file_hash(f2)