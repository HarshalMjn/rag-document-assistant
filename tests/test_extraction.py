import codecs

import pytest

from app.services.extraction_service import ExtractionError, ExtractionService
from tests.conftest import make_pdf


def test_pdf_keeps_real_page_numbers_and_skips_blank_pages():
    data = make_pdf(["Leave policy: 15 days", "", "Remote work: 2 days"])

    pages = ExtractionService.extract(data, "policy.PDF")

    assert [(p.page, p.text) for p in pages] == [
        (1, "Leave policy: 15 days"),
        (3, "Remote work: 2 days"),
    ]


def test_txt_is_normalized_and_has_no_page():
    data = "﻿Line 1\r\n\r\n\r\n\r\nLine 2   \r\n".encode()

    pages = ExtractionService.extract(data, "notes.txt")

    assert [(p.page, p.text) for p in pages] == [(None, "Line 1\n\nLine 2")]


@pytest.mark.parametrize(
    ("data", "filename", "message"),
    [
        (make_pdf(["", ""]), "scan.pdf", "OCR is not supported"),
        (b"hello", "fake.pdf", "not a valid PDF"),
        (b"  \n ", "empty.txt", "The file is empty"),
        (
            codecs.BOM_UTF16_LE + "hi".encode("utf-16-le"),
            "u.txt",
            "this file is UTF-16",
        ),
        ("café".encode("latin-1"), "l.txt", "must be UTF-8"),
        (b"abc\x00def", "b.txt", "looks binary"),
        (b"PK", "a.docx", "Unsupported file type '.docx'"),
    ],
)
def test_unreadable_files_raise_clear_errors(data, filename, message):
    with pytest.raises(ExtractionError, match=message):
        ExtractionService.extract(data, filename)
