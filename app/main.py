import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.config.database import engine
from app.routes import chat, config, documents, health
from app.services.qdrant_service import ensure_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await run_in_threadpool(ensure_collection)
    except Exception:
        logger.exception("Could not ensure Qdrant collection on startup")
    yield
    await engine.dispose()


app = FastAPI(
    title="RAG Document Assistant",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(config.router)
app.include_router(documents.router)
app.include_router(chat.router)
