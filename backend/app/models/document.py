import enum

from sqlalchemy import Column, DateTime, ForeignKey, Integer, func
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    active_version_id = Column(
        Integer,
        ForeignKey(
            "document_versions.id",
            name="fk_documents_active_version_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
    )
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime, nullable=True, index=True)

    versions = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version",
        foreign_keys="DocumentVersion.document_id",
    )

    @property
    def active(self):
        return next((v for v in self.versions if v.id == self.active_version_id), None)

    @property
    def latest(self):
        return self.versions[-1] if self.versions else None

    @property
    def current(self):
        return self.active or self.latest

    @property
    def active_version(self) -> int | None:
        return self.active.version if self.active else None

    @property
    def status(self) -> DocumentStatus:
        if self.active:
            return DocumentStatus.INDEXED
        return self.latest.status if self.latest else DocumentStatus.UPLOADED

    @property
    def filename(self) -> str | None:
        return self.current.filename if self.current else None

    @property
    def source_type(self) -> SourceType:
        return self.current.source_type if self.current else SourceType.FILE

    @property
    def source_url(self) -> str | None:
        return self.current.source_url if self.current else None
