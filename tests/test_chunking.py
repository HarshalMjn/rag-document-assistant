import pytest

from app.services.chunking_service import ChunkingService
from app.services.extraction_service import ExtractedPage

SENTENCES = " ".join(
    f"Sentence number {i} explains part of the leave policy." for i in range(40)
)


def test_short_page_is_one_chunk():
    chunks = ChunkingService.chunk_pages([ExtractedPage(1, "Employees get 15 days.")])

    assert [(c.chunk_index, c.page, c.text) for c in chunks] == [
        (0, 1, "Employees get 15 days.")
    ]


def test_chunks_respect_size_and_end_on_sentences():
    chunks = ChunkingService.chunk_pages(
        [ExtractedPage(2, SENTENCES)], chunk_size=800, chunk_overlap=120
    )

    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = ChunkingService.chunk_pages(
        [ExtractedPage(1, SENTENCES)], chunk_size=300, chunk_overlap=100
    )

    for previous, current in zip(chunks, chunks[1:], strict=False):
        # The next chunk starts with a sentence the previous one ended with.
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text


def test_chunks_never_cross_pages_and_indexes_are_global():
    pages = [ExtractedPage(1, SENTENCES), ExtractedPage(2, "Page two text.")]

    chunks = ChunkingService.chunk_pages(pages, chunk_size=500, chunk_overlap=50)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2
    assert chunks[-1].text == "Page two text."
    assert all("Page two" not in c.text for c in chunks if c.page == 1)


def test_text_without_spaces_is_hard_cut():
    chunks = ChunkingService.chunk_pages(
        [ExtractedPage(None, "x" * 2000)], chunk_size=800, chunk_overlap=0
    )

    assert [len(c.text) for c in chunks] == [800, 800, 400]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        ChunkingService.chunk_pages(
            [ExtractedPage(1, "x")], chunk_size=100, chunk_overlap=100
        )
