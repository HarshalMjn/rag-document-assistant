from dataclasses import dataclass

from app.config.settings import settings
from app.services.extraction_service import ExtractedPage

# Tried in order: paragraphs, lines, sentences, words, then a hard cut.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    # Position across the whole document, so chunks keep a stable order.
    chunk_index: int
    page: int | None
    text: str


class ChunkingService:
    @classmethod
    def chunk_pages(
        cls,
        pages: list[ExtractedPage],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Chunk]:
        """Split pages into overlapping chunks of at most `chunk_size` characters.

        Chunks never cross a page boundary, so each one cites exactly one page.
        Size and overlap default to settings; passing them lets evaluation compare configs.
        """
        size = chunk_size or settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        if overlap >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks: list[Chunk] = []
        for page in pages:
            pieces = cls._split(page.text, _SEPARATORS, size)
            for text in cls._merge(pieces, size, overlap):
                chunks.append(Chunk(chunk_index=len(chunks), page=page.page, text=text))
        return chunks

    @classmethod
    def _split(cls, text: str, separators: list[str], size: int) -> list[str]:
        """Break text into pieces no longer than `size`, using the coarsest separator possible."""
        sep = next(s for s in separators if s == "" or s in text)
        if sep == "":
            return [text[i : i + size] for i in range(0, len(text), size)]

        # Keep the separator on the end of each part, so joining pieces restores the text.
        parts = text.split(sep)
        parts = [p + sep for p in parts[:-1]] + [parts[-1]]

        finer = separators[separators.index(sep) + 1 :]
        pieces: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) <= size:
                pieces.append(part)
            else:
                pieces.extend(cls._split(part, finer, size))
        return pieces

    @staticmethod
    def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
        """Join small pieces into chunks, carrying the last ~`overlap` characters forward."""
        chunks: list[str] = []
        window: list[str] = []
        length = 0

        for piece in pieces:
            if window and length + len(piece) > size:
                chunks.append("".join(window).strip())
                # Drop pieces from the front until what's left fits the overlap and the new piece.
                while window and (length > overlap or length + len(piece) > size):
                    length -= len(window.pop(0))
            window.append(piece)
            length += len(piece)

        if window:
            chunks.append("".join(window).strip())
        return [c for c in chunks if c]
