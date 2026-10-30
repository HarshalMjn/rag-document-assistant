import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.clients.vector_store_client import ensure_collection
from app.config.database import SessionLocal, engine
from app.routes import chat, config, documents, health
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await run_in_threadpool(ensure_collection)
    except Exception:
        logger.exception("Could not ensure Qdrant collection on startup")
    try:
        async with SessionLocal() as db:
            await IngestionService.recover_interrupted(db)
    except Exception:
        logger.exception("Could not recover interrupted uploads on startup")
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
