from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.database import get_db
from app.controllers.document_controller import DocumentController
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentOut,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        200: {
            "model": DocumentUploadResponse,
            "description": "Duplicate: already indexed",
        },
        422: {"model": DocumentUploadResponse, "description": "Indexing failed"},
        400: {
            "description": "Missing name, unsupported type, empty file or invalid PDF"
        },
        413: {"description": "File larger than MAX_UPLOAD_MB"},
    },
)
async def upload_document(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    return await DocumentController.upload(db, file)


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    return await DocumentController.list_documents(db)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    return await DocumentController.get_document(db, document_id)


@router.put(
    "/{document_id}",
    response_model=DocumentUploadResponse,
    responses={
        200: {"description": "New version indexed, or unchanged content (duplicate)"},
        404: {"description": "Document not found"},
        409: {"description": "Another update of this document is in progress"},
        422: {
            "model": DocumentUploadResponse,
            "description": "Indexing failed; the document stays on its old version",
        },
        400: {
            "description": "Missing name, unsupported type, empty file or invalid PDF"
        },
        413: {"description": "File larger than MAX_UPLOAD_MB"},
    },
)
async def update_document(
    document_id: str, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    return await DocumentController.update(db, document_id, file)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={404: {"description": "Document not found or already deleted"}},
)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    return await DocumentController.delete(db, document_id)
