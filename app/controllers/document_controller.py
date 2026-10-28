import logging
from pathlib import PurePath

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config.settings import settings
from app.models.document import Document, DocumentStatus
from app.schemas.document import DocumentOut, DocumentUploadResponse
from app.services.extraction_service import SUPPORTED_EXTENSIONS
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


class DocumentController:
    @classmethod
    async def upload(cls, db: AsyncSession, file: UploadFile):
        # Browsers may send a full client path; keep only the file name.
        filename = PurePath((file.filename or "").replace("\\", "/")).name[:255]
        data = await cls._read_validated(file, filename)
        document, duplicate = await IngestionService.ingest_file(
            db, data, filename, file.content_type
        )

        body = DocumentUploadResponse(
            document=DocumentOut.model_validate(document), duplicate=duplicate
        )
        if duplicate:
            status_code = 200
        elif document.status == DocumentStatus.INDEXED:
            status_code = 201
        else:
            # Indexing failed: the body still carries the document id and the reason.
            status_code = 422
        return JSONResponse(
            status_code=status_code, content=body.model_dump(mode="json")
        )

    @classmethod
    async def _read_validated(cls, file: UploadFile, filename: str) -> bytes:
        """Reject bad uploads before any document row is created."""
        if not filename:
            raise HTTPException(status_code=400, detail="File name is missing")

        extension = PurePath(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{extension or 'none'}'. Supported: .pdf, .txt",
            )

        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        # Read one byte past the limit, so oversized files are caught without loading them whole.
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than the {settings.MAX_UPLOAD_MB} MB limit",
            )
        if not data:
            raise HTTPException(status_code=400, detail="The file is empty")
        # The extension alone can lie; every PDF starts with this signature.
        if extension == ".pdf" and not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The file is not a valid PDF")
        return data

    @classmethod
    async def list_documents(cls, db: AsyncSession) -> list[DocumentOut]:
        result = await db.execute(
            select(Document)
            .options(selectinload(Document.versions))
            .where(Document.deleted_at.is_(None))
            .order_by(Document.created_at.desc())
        )
        return [DocumentOut.model_validate(d) for d in result.scalars().all()]

    @classmethod
    async def get_document(cls, db: AsyncSession, document_id: str) -> DocumentOut:
        document = await IngestionService.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentOut.model_validate(document)
