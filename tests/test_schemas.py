from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.document import Document, DocumentStatus, SourceType
from app.models.document_version import DocumentVersion
from app.models.message import Message, MessageRole
from app.schemas.chat import ChatRequest, MessageOut
from app.schemas.chunk import ChunkPayload
from app.schemas.document import DocumentOut

NOW = datetime(2026, 1, 1, 12, 0, 0)


def test_document_out_from_model():
    version = DocumentVersion(
        version=1,
        content_hash="a" * 64,
        status=DocumentStatus.INDEXED,
        chunk_count=3,
        created_at=NOW,
        updated_at=NOW,
    )
    document = Document(
        id="doc-1",
        filename="leave_policy.txt",
        source_type=SourceType.FILE,
        status=DocumentStatus.INDEXED,
        active_version=1,
        created_at=NOW,
        updated_at=NOW,
        versions=[version],
    )

    out = DocumentOut.model_validate(document)

    assert out.id == "doc-1"
    assert out.status == "indexed"
    assert out.active_version == 1
    assert out.versions[0].chunk_count == 3


def test_message_out_from_model():
    message = Message(
        id=7,
        role=MessageRole.ASSISTANT,
        content="Annual leave is 20 days.",
        citations=[{"document_id": "doc-1", "filename": "leave_policy.txt", "page": 1, "chunk_id": "c1"}],
        created_at=NOW,
    )

    out = MessageOut.model_validate(message)

    assert out.role == "assistant"
    assert out.citations[0].filename == "leave_policy.txt"


def test_chunk_payload_round_trips_through_qdrant_dict():
    payload = ChunkPayload(
        document_id="doc-1",
        version=2,
        chunk_id="doc-1:2:0",
        chunk_index=0,
        filename="leave_policy.txt",
        content_hash="b" * 64,
        text="Annual leave is 20 days.",
    )

    # Qdrant stores payloads as plain dicts.
    restored = ChunkPayload.model_validate(payload.model_dump())

    assert restored == payload
    assert restored.page is None


def test_chat_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="")
