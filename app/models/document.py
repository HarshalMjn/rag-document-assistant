import enum
import uuid

from sqlalchemy import Column, DateTime, Enum, Integer, String, func
from sqlalchemy.orm import relationship

from app.config.database import Base


class DocumentStatus(enum.StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class SourceType(enum.StrEnum):
    FILE = "file"
    URL = "url"


class Document(Base):
    __tablename__ = "documents"

    # Stable document_id: stays the same across updates, so PUT/DELETE target the exact document.
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    source_type = Column(Enum(SourceType), nullable=False, default=SourceType.FILE)
    source_url = Column(String(2048), nullable=True)
    status = Column(
        Enum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADED
    )
    # Only chunks of this version are retrievable; null until the first version is indexed.
    active_version = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime, nullable=True, index=True)

    # Relationships
    versions = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version",
    )
