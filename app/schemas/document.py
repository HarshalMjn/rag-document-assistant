from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus, SourceType


class DocumentVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: int
    content_hash: str
    status: DocumentStatus
    error: str | None = None
    mime_type: str | None = None
    file_size: int | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    source_type: SourceType
    source_url: str | None = None
    status: DocumentStatus
    active_version: int | None = None
    created_at: datetime
    updated_at: datetime
    versions: list[DocumentVersionOut] = []


class DocumentUploadResponse(BaseModel):
    document: DocumentOut
    # True when the uploaded bytes matched an already indexed version, so nothing was re-indexed.
    duplicate: bool = False


class DocumentDeleteResponse(BaseModel):
    id: str
    deleted: bool
    chunks_removed: int
