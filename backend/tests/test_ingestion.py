"""Requirement 3A: document upload and ingestion.

Checks what test_scenarios.py does not: the metadata stored with each chunk, configurable
chunking, batched embeddings, honest status reporting and the remaining unreadable inputs.
Runs with fake providers, so it needs no API keys.
"""

import codecs
import hashlib
import math
import uuid
from itertools import pairwise

import pymupdf
import pytest
from conftest import make_pdf, sample, unique, upload, upload_ok
from qdrant_client import models
from sqlalchemy import select

from app.clients import vector_store_client
from app.clients.embedding_client import EmbeddingClient, EmbeddingClientError
from app.config.database import SessionLocal
from app.config.settings import settings
from app.models import Document
from app.models.document_version import DocumentVersion
from app.schemas.chunk import ChunkPayload
from app.services.chunking_service import ChunkingService
from app.services.extraction_service import ExtractedPage

# 30 short sentences joined by ". ", so chunks split cleanly on sentence boundaries.
SENTENCES = " ".join(f"Sentence {i} is about annual leave." for i in range(30))


def stored_chunks(document_id: int) -> list[ChunkPayload]:
    """Every chunk Qdrant holds for a document, in chunk order."""
    points, _ = vector_store_client.qdrant_client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                )
            ]
        ),
        limit=1000,
        with_payload=True,
    )
    chunks = [ChunkPayload.model_validate(point.payload) for point in points]
    return sorted(chunks, key=lambda chunk: chunk.chunk_index)


def first_sentence(text: str) -> str:
    return text.split(". ")[0]


# Metadata


async def test_chunks_store_document_metadata(client):
    data = unique(sample("leave_policy_v1.txt"))

    document = await upload_ok(client, "leave_policy_v1.txt", data)

    version = document["versions"][0]
    assert version["status"] == "indexed"
    assert version["content_hash"] == hashlib.sha256(data).hexdigest()
    assert version["file_size"] == len(data)
    assert version["mime_type"] == "text/plain"
    assert document["created_at"] and document["updated_at"]

    chunks = stored_chunks(document["id"])
    assert len(chunks) == version["chunk_count"] > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.document_id == document["id"]
        assert chunk.version == 1
        assert chunk.filename == "leave_policy_v1.txt"
        assert chunk.page is None  # TXT files have no pages
        assert chunk.content_hash == version["content_hash"]
        assert chunk.chunk_id == vector_store_client.make_chunk_id(
            document["id"], 1, chunk.chunk_index
        )


async def test_pdf_chunks_keep_page_numbers(client):
    token = uuid.uuid4().hex
    data = make_pdf(
        f"Annual leave is 20 days. Ref {token}",
        f"Sick leave is 10 days. Ref {token}",
    )

    document = await upload_ok(client, "policy.pdf", data, "application/pdf")

    assert document["versions"][0]["page_count"] == 2
    chunks = stored_chunks(document["id"])
    assert [c.page for c in chunks] == [1, 2]
    # A chunk never crosses a page boundary.
    assert "Annual leave" in chunks[0].text and "Sick leave" not in chunks[0].text
    assert "Sick leave" in chunks[1].text and "Annual leave" not in chunks[1].text


async def test_list_documents_shows_metadata(client):
    document = await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )

    response = await client.get("/documents")

    assert response.status_code == 200, response.text
    [listed] = [d for d in response.json()["data"] if d["id"] == document["id"]]
    assert listed["filename"] == "leave_policy_v1.txt"
    assert listed["status"] == "indexed"
    assert listed["active_version"] == 1
    assert listed["created_at"] and listed["updated_at"]
    assert listed["versions"][0]["chunk_count"] >= 1


# Chunking


async def test_chunk_size_and_overlap_come_from_settings(client, monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE", 200)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 80)

    document = await upload_ok(client, "sentences.txt", unique(SENTENCES.encode()))

    texts = [c.text for c in stored_chunks(document["id"])]
    assert len(texts) > 3
    assert all(len(text) <= 200 for text in texts)
    # With overlap, each chunk starts with a sentence the previous chunk ended with.
    for previous, current in pairwise(texts):
        assert first_sentence(current) in previous


def test_zero_overlap_shares_no_text():
    pages = [ExtractedPage(page=None, text=SENTENCES)]

    chunks = ChunkingService.chunk_pages(pages, chunk_size=200, chunk_overlap=0)

    texts = [c.text for c in chunks]
    assert len(texts) > 3
    for previous, current in pairwise(texts):
        assert first_sentence(current) not in previous


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkingService.chunk_pages(
            [ExtractedPage(page=None, text=SENTENCES)],
            chunk_size=100,
            chunk_overlap=100,
        )


# Embedding batches


async def test_embeddings_are_sent_in_batches(client, fake_providers, monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(settings, "CHUNK_SIZE", 200)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 0)

    document = await upload_ok(client, "sentences.txt", unique(SENTENCES.encode()))

    chunk_count = document["versions"][0]["chunk_count"]
    sizes = [len(batch) for batch in fake_providers.embed_batches]
    assert chunk_count >= 3
    assert len(sizes) == math.ceil(chunk_count / 2)
    assert all(size == 2 for size in sizes[:-1])
    assert sum(sizes) == chunk_count


# Status


async def test_version_is_not_indexed_until_chunks_are_stored(
    client, fake_providers, monkeypatch
):
    data = unique(sample("leave_policy_v1.txt"))
    content_hash = hashlib.sha256(data).hexdigest()
    seen = []

    async def embed_and_check(texts, task_type):
        # Look at MySQL while indexing is still running.
        async with SessionLocal() as db:
            version = (
                await db.execute(
                    select(DocumentVersion).where(
                        DocumentVersion.content_hash == content_hash
                    )
                )
            ).scalar_one()
            document = await db.get(Document, version.document_id)
            seen.append((version.status, document.active_version_id))
        return await fake_providers.embed(texts, task_type)

    monkeypatch.setattr(EmbeddingClient, "_embed", embed_and_check)

    document = await upload_ok(client, "leave_policy_v1.txt", data)

    # While embedding: still processing and not live. After: indexed and live.
    assert seen and all(seen_item == ("processing", None) for seen_item in seen)
    assert document["status"] == "indexed"
    assert document["active_version"] == 1


async def test_failed_upload_can_be_retried(client, fake_providers):
    data = unique(sample("leave_policy_v1.txt"))
    fake_providers.embed_error = EmbeddingClientError("Gemini quota exceeded")
    failed = await upload(client, "leave_policy_v1.txt", data)
    assert failed.status_code == 422, failed.text

    fake_providers.embed_error = None
    retried = await upload(client, "leave_policy_v1.txt", data)

    # A failed version is not a duplicate, so the same bytes are indexed this time.
    assert retried.status_code == 201, retried.text
    assert retried.json()["data"]["duplicate"] is False
    assert retried.json()["data"]["document"]["status"] == "indexed"


# Unreadable inputs


async def test_file_over_size_limit_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)

    response = await upload(client, "big.txt", b"a" * (1024 * 1024 + 1))

    assert response.status_code == 413, response.text
    assert "larger than the 1 MB limit" in response.json()["detail"]


def _encrypted_pdf() -> bytes:
    document = pymupdf.open(stream=make_pdf("Annual leave is 20 days."))
    return document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret"
    )


@pytest.mark.parametrize(
    ("filename", "data", "error"),
    [
        (
            "utf16.txt",
            codecs.BOM_UTF16_LE + "Annual leave is 20 days.".encode("utf-16-le"),
            "this file is UTF-16",
        ),
        ("latin1.txt", "Café annual leave policy".encode("latin-1"), "must be UTF-8"),
        ("binary.txt", b"annual\x00leave\x00policy", "looks binary"),
        ("locked.pdf", _encrypted_pdf(), "password-protected"),
        ("corrupt.pdf", b"%PDF-1.7\nthis is not really a PDF", "not a valid PDF"),
    ],
    ids=["utf16", "latin1", "binary", "password-pdf", "corrupt-pdf"],
)
async def test_unreadable_file_fails_with_reason(
    client, fake_providers, filename, data, error
):
    content_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"

    response = await upload(client, filename, data, content_type)

    assert response.status_code == 422, response.text
    document = response.json()["data"]["document"]
    assert document["status"] == "failed"
    assert document["active_version"] is None
    assert error in document["versions"][-1]["error"]
    assert fake_providers.embed_batches == []
    assert vector_store_client.count_chunks(document["id"]) == 0
