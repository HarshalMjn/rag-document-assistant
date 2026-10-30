import logging

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.clients.vector_store_client import qdrant_client
from app.config.database import engine

logger = logging.getLogger(__name__)


class HealthController:
    @classmethod
    async def health(cls):
        checks = {"qdrant": "ok", "mysql": "ok"}

        try:
            await run_in_threadpool(qdrant_client.get_collections)
        except Exception:
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
            content={"data": {"status": "ok" if healthy else "degraded", **checks}},
        )
