import logging

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config.database import engine
from app.config.settings import settings
from app.services.qdrant_service import qdrant_client

logger = logging.getLogger(__name__)


class HealthController:
    @classmethod
    def config(cls) -> dict:
        """Non-secret settings the UI shows. Keys, hosts and passwords are never included."""
        return {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K,
            "min_score": settings.MIN_SCORE,
            "history_limit": settings.HISTORY_LIMIT,
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "rewrite_model": settings.REWRITE_MODEL,
        }

    @classmethod
    async def health(cls):
        checks = {"qdrant": "ok", "mysql": "ok"}

        try:
            await run_in_threadpool(qdrant_client.get_collections)
        except Exception:
            # Log details server-side only; the response never includes connection info.
            logger.exception("Qdrant health check failed")
            checks["qdrant"] = "error"

        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            logger.exception("MySQL health check failed")
            checks["mysql"] = "error"

        healthy = all(value == "ok" for value in checks.values())
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "degraded", **checks},
        )
