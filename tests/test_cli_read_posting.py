import pytest

from assay.cli import PostingError, read_posting


def test_reads_a_file(posting_file):
    path = posting_file("Some text.")
    assert read_posting(path) == ("Some text.", path)


def test_reads_the_clipboard_without_a_path(clipboard):
    clipboard("From clipboard.")
    assert read_posting(None) == ("From clipboard.", "clipboard")


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(PostingError, match="Could not read"):
        read_posting(str(tmp_path / "nope.txt"))


def test_file_that_is_not_utf8_is_an_error(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(PostingError, match="Could not read"):
        read_posting(str(path))


def test_empty_file_is_an_error(posting_file):
    with pytest.raises(PostingError, match="Nothing to read"):
        read_posting(posting_file("  \n"))


def test_empty_clipboard_is_an_error():
    with pytest.raises(PostingError, match="Nothing to read"):
        read_posting(None)
