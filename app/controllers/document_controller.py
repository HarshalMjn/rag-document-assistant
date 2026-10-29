import logging
from pathlib import PurePath

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config.settings import settings
from app.models.document import Document, DocumentStatus
from app.schemas.common import DataResponse
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentOut,
    DocumentUploadResponse,
)
from app.services.extraction_service import SUPPORTED_EXTENSIONS
from app.services.ingestion_service import IngestionService, UpdateConflictError

logger = logging.getLogger(__name__)


class DocumentController:
    @classmethod
    async def upload(cls, db: AsyncSession, file: UploadFile):
        filename = cls._clean_filename(file)
        data = await cls._read_validated(file, filename)
        document, duplicate = await IngestionService.ingest_file(
            db, data, filename, file.content_type
        )
        return cls._upload_response(document, duplicate, success_status=201)

    @classmethod
    async def update(cls, db: AsyncSession, document_id: int, file: UploadFile):
        document = await IngestionService.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        filename = cls._clean_filename(file)
        data = await cls._read_validated(file, filename)
        try:
            document, duplicate = await IngestionService.update_file(
                db, document, data, filename, file.content_type
            )
        except UpdateConflictError as e:
            raise HTTPException(
                status_code=409,
                detail="Another update of this document is in progress; try again",
            ) from e
        return cls._upload_response(document, duplicate, success_status=200)

    @staticmethod
    def _clean_filename(file: UploadFile) -> str:
        return PurePath((file.filename or "").replace("\\", "/")).name[:255]

    @staticmethod
    def _upload_response(
        document: Document, duplicate: bool, success_status: int
    ) -> JSONResponse:
        """200 for a duplicate, `success_status` once indexed, 422 if indexing failed."""
        body = DocumentUploadResponse(
            document=DocumentOut.model_validate(document), duplicate=duplicate
        )
        # An update can fail while the document stays indexed on its old version, so the
        # newest version's status decides, not the document's.
        latest = document.versions[-1]
        if duplicate:
            status_code = 200
        elif latest.status == DocumentStatus.INDEXED:
            status_code = success_status
        else:
            # Indexing failed: the body still carries the document id and the reason.
            status_code = 422
        return JSONResponse(
            status_code=status_code, content={"data": body.model_dump(mode="json")}
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
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than the {settings.MAX_UPLOAD_MB} MB limit",
            )
        if not data:
            raise HTTPException(status_code=400, detail="The file is empty")

        if extension == ".pdf" and not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The file is not a valid PDF")
        return data

    @classmethod
    async def list_documents(cls, db: AsyncSession) -> DataResponse[list[DocumentOut]]:
        result = await db.execute(
            select(Document)
            .options(selectinload(Document.versions))
            .where(Document.deleted_at.is_(None))
            .order_by(Document.created_at.desc())
        )
        return DataResponse(
            data=[DocumentOut.model_validate(d) for d in result.scalars().all()]
        )

    @classmethod
    async def get_document(
        cls, db: AsyncSession, document_id: int
    ) -> DataResponse[DocumentOut]:
        document = await IngestionService.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DataResponse(data=DocumentOut.model_validate(document))

    @classmethod
    async def delete(
        cls, db: AsyncSession, document_id: int
    ) -> DataResponse[DocumentDeleteResponse]:
        document = await IngestionService.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        removed = await IngestionService.delete_document(db, document)
        return DataResponse(
            data=DocumentDeleteResponse(
                id=document_id, deleted=True, chunks_removed=removed
            )
        )
