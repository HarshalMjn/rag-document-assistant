from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config.settings import settings
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.services import qdrant_service
from app.services.gemini_service import GeminiService
from app.services.qdrant_service import SearchHit


@dataclass(frozen=True)
class RetrievalResult:
    # Every chunk Qdrant returned, best first; kept for the retrieval trace.
    hits: list[SearchHit]
    # The subset scoring at least MIN_SCORE: the only chunks allowed as evidence.
    evidence: list[SearchHit]


class RetrievalService:
    @classmethod
    async def retrieve(
        cls,
        db: AsyncSession,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Find the chunks most relevant to `query` across all live document versions.

        `top_k` and `min_score` default to settings; passing them lets evaluation compare configs.
        """
        top_k = top_k or settings.TOP_K
        min_score = settings.MIN_SCORE if min_score is None else min_score

        active_versions = await cls.active_versions(db)
        if not active_versions:
            # Nothing is indexed, so there is no need to embed the question.
            return RetrievalResult(hits=[], evidence=[])

        vector = await GeminiService.embed_query(query)
        hits = await run_in_threadpool(
            qdrant_service.search, vector, active_versions, top_k
        )
        evidence = [hit for hit in hits if hit.score >= min_score]
        return RetrievalResult(hits=hits, evidence=evidence)

    @classmethod
    async def active_versions(cls, db: AsyncSession) -> list[tuple[int, int]]:
        """(document_id, active_version) for every non-deleted document that has a live version.

        MySQL is the source of truth for what is live; Qdrant only stores the chunks.
        """
        # Qdrant payloads carry the version number, so map the active version row to it.
        result = await db.execute(
            select(Document.id, DocumentVersion.version)
            .join(DocumentVersion, DocumentVersion.id == Document.active_version_id)
            .where(Document.deleted_at.is_(None))
        )
        return [(document_id, version) for document_id, version in result.all()]
