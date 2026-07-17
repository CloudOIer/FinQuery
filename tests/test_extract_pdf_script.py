from scripts.extract_pdf import iter_pdfs, progress_bar


def test_progress_bar_formats_file_progress():
    assert progress_bar(0, 4) == "[----------------------------] 0/4 0%"
    assert progress_bar(2, 4) == "[##############--------------] 2/4 50%"
    assert progress_bar(4, 4) == "[############################] 4/4 100%"


def test_iter_pdfs_discovers_directory_and_respects_limit(tmp_path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "nested" / "b.pdf"
    ignored = tmp_path / "note.txt"
    second.parent.mkdir()
    first.write_bytes(b"pdf")
    second.write_bytes(b"pdf")
    ignored.write_text("not pdf", encoding="utf-8")

    assert iter_pdfs([str(tmp_path)], limit=1) == [first]
    assert iter_pdfs([str(tmp_path)]) == [first, second]
