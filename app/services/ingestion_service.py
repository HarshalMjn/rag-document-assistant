import hashlib
import logging

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from app.models.document import Document, DocumentStatus, SourceType
from app.models.document_version import DocumentVersion
from app.schemas.chunk import ChunkPayload
from app.services import qdrant_service
from app.services.chunking_service import ChunkingService
from app.services.extraction_service import ExtractionError, ExtractionService
from app.services.gemini_service import GeminiService, GeminiServiceError

logger = logging.getLogger(__name__)


class IngestionService:
    @classmethod
    async def ingest_file(
        cls,
        db: AsyncSession,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
    ) -> tuple[Document, bool]:
        """Create a document and index it as version 1. Returns (document, duplicate).

        If the same bytes are already indexed, the existing document is returned with
        duplicate=True and nothing is re-embedded. Otherwise this returns once indexing
        has finished, so the status is final (`indexed` or `failed`, with the reason on
        the version's `error`).
        """
        content_hash = hashlib.sha256(data).hexdigest()
        existing = await cls.find_indexed_by_hash(db, content_hash)
        if existing is not None:
            logger.info("Duplicate upload of document %s", existing.id)
            return existing, True

        document = Document(
            filename=filename,
            source_type=SourceType.FILE,
            status=DocumentStatus.UPLOADED,
        )
        version = DocumentVersion(
            document=document,
            version=1,
            content_hash=content_hash,
            status=DocumentStatus.UPLOADED,
            mime_type=mime_type,
            file_size=len(data),
        )
        db.add_all([document, version])
        await db.commit()

        await cls.index_version(db, document, version, data)
        return await cls.get_document(db, document.id), False

    @classmethod
    async def index_version(
        cls,
        db: AsyncSession,
        document: Document,
        version: DocumentVersion,
        data: bytes,
    ) -> None:
        """Extract, chunk, embed and store one version, then make it the active version.

        On failure the version is marked failed and its partial chunks are removed.
        A previously active version is left untouched, so updates (Phase 5) can reuse this.
        """
        version.status = DocumentStatus.PROCESSING
        if document.active_version is None:
            document.status = DocumentStatus.PROCESSING
        await db.commit()

        try:
            chunk_count, page_count = await cls._index(document, version, data)
        except Exception as e:
            await cls._mark_failed(db, document, version, e)
            return

        version.status = DocumentStatus.INDEXED
        version.error = None
        version.chunk_count = chunk_count
        version.page_count = page_count
        # Switched only after every chunk is stored, so retrieval never sees a half-indexed version.
        document.active_version = version.version
        document.status = DocumentStatus.INDEXED
        await db.commit()
        logger.info(
            "Indexed document %s v%s: %s chunks",
            document.id,
            version.version,
            chunk_count,
        )

    @classmethod
    async def find_indexed_by_hash(
        cls, db: AsyncSession, content_hash: str
    ) -> Document | None:
        """Find a non-deleted document whose active version has these exact bytes.

        Failed uploads don't count, so re-uploading a file that failed indexes it again.
        """
        result = await db.execute(
            select(Document.id)
            .join(
                DocumentVersion,
                and_(
                    DocumentVersion.document_id == Document.id,
                    DocumentVersion.version == Document.active_version,
                ),
            )
            .where(
                DocumentVersion.content_hash == content_hash,
                Document.deleted_at.is_(None),
            )
            .limit(1)
        )
        document_id = result.scalar_one_or_none()
        return await cls.get_document(db, document_id) if document_id else None

    @classmethod
    async def get_document(cls, db: AsyncSession, document_id: str) -> Document | None:
        """Load a non-deleted document with its versions."""
        result = await db.execute(
            select(Document)
            .options(selectinload(Document.versions))
            .where(Document.id == document_id, Document.deleted_at.is_(None))
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    @classmethod
    async def _index(
        cls, document: Document, version: DocumentVersion, data: bytes
    ) -> tuple[int, int]:
        """Run the pipeline. Returns (chunk_count, pages with text)."""
        # Parsing and Qdrant calls are blocking, so they run in a thread pool.
        pages = await run_in_threadpool(
            ExtractionService.extract, data, document.filename
        )
        chunks = ChunkingService.chunk_pages(pages)

        payloads = [
            ChunkPayload(
                document_id=document.id,
                version=version.version,
                chunk_id=qdrant_service.make_chunk_id(
                    document.id, version.version, chunk.chunk_index
                ),
                chunk_index=chunk.chunk_index,
                filename=document.filename,
                page=chunk.page,
                content_hash=version.content_hash,
                source_url=document.source_url,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        vectors = await GeminiService.embed_documents([p.text for p in payloads])
        await run_in_threadpool(qdrant_service.upsert_chunks, payloads, vectors)

        stored = await run_in_threadpool(
            qdrant_service.count_chunks, document.id, version.version
        )
        if stored != len(payloads):
            raise RuntimeError(
                f"Qdrant stored {stored} of {len(payloads)} chunks for document {document.id}"
            )
        return len(payloads), len(pages)

    @classmethod
    async def _mark_failed(
        cls,
        db: AsyncSession,
        document: Document,
        version: DocumentVersion,
        error: Exception,
    ) -> None:
        if isinstance(error, ExtractionError | GeminiServiceError):
            # These messages are written to be safe for users.
            reason = str(error)
            logger.warning(
                "Indexing failed for document %s v%s: %s",
                document.id,
                version.version,
                reason,
            )
        else:
            reason = "Indexing failed due to an internal error"
            logger.exception(
                "Indexing failed for document %s v%s", document.id, version.version
            )

        # Remove any chunks written before the failure; a failed version must not be searchable.
        try:
            await run_in_threadpool(
                qdrant_service.delete_version, document.id, version.version
            )
        except Exception:
            logger.exception(
                "Could not clean up chunks for document %s v%s",
                document.id,
                version.version,
            )

        version.status = DocumentStatus.FAILED
        version.error = reason
        # With an older active version (an update), the document stays indexed on that version.
        if document.active_version is None:
            document.status = DocumentStatus.FAILED
        await db.commit()
