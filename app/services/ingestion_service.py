import hashlib
import logging

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
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


class UpdateConflictError(Exception):
    """Another update of the same document claimed the next version number first."""


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

        document = Document()
        version = DocumentVersion(
            document=document,
            version=1,
            content_hash=content_hash,
            status=DocumentStatus.UPLOADED,
            filename=filename,
            source_type=SourceType.FILE,
            mime_type=mime_type,
            file_size=len(data),
        )
        db.add_all([document, version])
        await db.commit()

        await cls.index_version(db, document, version, data)
        return await cls.get_document(db, document.id), False

    @classmethod
    async def update_file(
        cls,
        db: AsyncSession,
        document: Document,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
    ) -> tuple[Document, bool]:
        """Index new bytes as the next version of `document`. Returns (document, duplicate).

        The old version stays live until the new one is fully indexed; only then are its
        chunks removed. If indexing fails the document stays on the old version and the
        new version is kept as failed, with the reason on its `error`. Bytes identical to
        the active version are a no-op (duplicate=True).
        """
        content_hash = hashlib.sha256(data).hexdigest()
        previous = document.active_version
        active = document.active
        if active is not None and active.content_hash == content_hash:
            logger.info("Update of document %s has unchanged content", document.id)
            return document, True

        # Failed versions keep their numbers, so the next one is always above all of them.
        number = max((v.version for v in document.versions), default=0) + 1
        version = DocumentVersion(
            document=document,
            version=number,
            content_hash=content_hash,
            status=DocumentStatus.UPLOADED,
            filename=filename,
            source_type=SourceType.FILE,
            mime_type=mime_type,
            file_size=len(data),
        )
        db.add(version)
        try:
            await db.commit()
        except IntegrityError as e:
            # A parallel update took this number (unique per document); let the caller retry.
            await db.rollback()
            raise UpdateConflictError(document.id) from e

        await cls.index_version(db, document, version, data)
        if version.status == DocumentStatus.INDEXED and previous is not None:
            await cls._remove_old_version(document.id, previous)
        return await cls.get_document(db, document.id), False

    @classmethod
    async def delete_document(cls, db: AsyncSession, document: Document) -> int:
        """Soft-delete a document and remove all its chunks. Returns the chunks removed.

        MySQL is marked first: retrieval reads live documents from MySQL, so the document
        stops being searchable at once, even if the Qdrant cleanup below fails (then 0 is
        returned and the orphaned chunks are never retrieved).
        """
        document.deleted_at = func.now()
        await db.commit()

        try:
            removed = await run_in_threadpool(qdrant_service.count_chunks, document.id)
            await run_in_threadpool(qdrant_service.delete_document, document.id)
        except Exception:
            logger.exception(
                "Could not remove chunks of deleted document %s", document.id
            )
            return 0
        logger.info("Deleted document %s: %s chunks removed", document.id, removed)
        return removed

    @classmethod
    async def _remove_old_version(cls, document_id: str, version: int) -> None:
        """Delete a replaced version's chunks. Failure is only logged: retrieval already
        ignores every version except the active one."""
        try:
            await run_in_threadpool(qdrant_service.delete_version, document_id, version)
        except Exception:
            logger.exception(
                "Could not remove old chunks of document %s v%s", document_id, version
            )

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
        A previously active version is left untouched, so a failed update keeps it live.
        """
        version.status = DocumentStatus.PROCESSING
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
        document.active_version_id = version.id
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
                    DocumentVersion.id == Document.active_version_id,
                ),
            )
            .where(
                DocumentVersion.content_hash == content_hash,
                Document.deleted_at.is_(None),
            )
            .limit(1)
        )
        document_id = result.scalar_one_or_none()
        return await cls.get_document(db, document_id) if document_id is not None else None

    @classmethod
    async def get_document(cls, db: AsyncSession, document_id: int) -> Document | None:
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
        pages = await run_in_threadpool(ExtractionService.extract, data, version.filename)
        chunks = ChunkingService.chunk_pages(pages)

        payloads = [
            ChunkPayload(
                document_id=document.id,
                version=version.version,
                chunk_id=qdrant_service.make_chunk_id(
                    document.id, version.version, chunk.chunk_index
                ),
                chunk_index=chunk.chunk_index,
                filename=version.filename,
                page=chunk.page,
                content_hash=version.content_hash,
                source_url=version.source_url,
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
        # active_version_id is untouched: after a failed update the old version stays live.
        await db.commit()
