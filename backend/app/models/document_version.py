from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from app.config.database import Base
from app.models.document import DocumentStatus, SourceType


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "version", name="uq_document_versions_doc_version"
        ),
        Index("ix_document_versions_content_hash", "content_hash"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    status = Column(
        Enum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADED
    )
    error = Column(Text, nullable=True)
    filename = Column(String(255), nullable=False)
    source_type = Column(Enum(SourceType), nullable=False, default=SourceType.FILE)
    source_url = Column(String(2048), nullable=True)
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    document = relationship(
        "Document", back_populates="versions", foreign_keys=[document_id]
    )
