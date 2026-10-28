import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.config.database import engine
from app.routes import health
from app.services.qdrant_service import ensure_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await run_in_threadpool(ensure_collection)
    except Exception:
        # Keep the API up so /health can report the problem instead of crashing on boot.
        logger.exception("Could not ensure Qdrant collection on startup")
    yield
    await engine.dispose()


app = FastAPI(
    title="RAG Document Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
