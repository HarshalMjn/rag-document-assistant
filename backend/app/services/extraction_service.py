import codecs
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

_EXTRA_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


class ExtractionError(Exception):
    """The file can't be turned into text. The message is safe to show to the user."""


@dataclass(frozen=True)
class ExtractedPage:
    page: int | None
    text: str


class ExtractionService:
    @classmethod
    def extract(cls, data: bytes, filename: str) -> list[ExtractedPage]:
        """Extract text page by page. Pages with no text are skipped.

        Raises ExtractionError when the file type is unsupported or no text is found.
        CPU-bound: call it from a thread pool inside async code.
        """
        extension = Path(filename).suffix.lower()
        if extension == ".pdf":
            pages = cls._extract_pdf(data)
        elif extension == ".txt":
            pages = cls._extract_txt(data)
        else:
            raise ExtractionError(
                f"Unsupported file type '{extension or 'none'}'. Supported: .pdf, .txt"
            )

        if not pages:
            if extension == ".pdf":
                raise ExtractionError(
                    "No extractable text found. The PDF may be scanned or image-only (OCR is not supported)"
                )
            raise ExtractionError("The file is empty")
        return pages

    @classmethod
    def _extract_pdf(cls, data: bytes) -> list[ExtractedPage]:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as e:
            logger.warning("Failed to open PDF: %s", e)
            raise ExtractionError("The file is not a valid PDF") from e

        with doc:
            if doc.needs_pass:
                raise ExtractionError("The PDF is password-protected")

            pages = []
            for index, page in enumerate(doc, start=1):
                text = cls._normalize(page.get_text("text"))
                if text:
                    pages.append(ExtractedPage(page=index, text=text))
            return pages

    @classmethod
    def _extract_txt(cls, data: bytes) -> list[ExtractedPage]:
        # Checked before the NUL-byte test, because UTF-16 text is full of NUL bytes.
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            raise ExtractionError(
                "Text files must be UTF-8 encoded (this file is UTF-16)"
            )
        try:
            # utf-8-sig also strips a BOM if present.
            raw = data.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ExtractionError("Text files must be UTF-8 encoded") from e
        if "\x00" in raw:
            raise ExtractionError("The file looks binary, not plain text")

        text = cls._normalize(raw)
        return [ExtractedPage(page=None, text=text)] if text else []

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _TRAILING_SPACES.sub("\n", text)
        text = _EXTRA_BLANK_LINES.sub("\n\n", text)
        return text.strip()
