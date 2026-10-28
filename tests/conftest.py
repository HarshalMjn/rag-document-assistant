import hashlib

import httpx
import pymupdf
import pytest
from sqlalchemy import delete

from app.config.database import SessionLocal, engine
from app.config.settings import settings
from app.main import app
from app.models import Document
from app.services import qdrant_service
from app.services.gemini_service import GeminiService

TEST_COLLECTION = "documents_test"


def fake_vector(text: str) -> list[float]:
    """Deterministic 768-dim vector from the text, so tests need no API key and cost nothing."""
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[i % len(digest)] / 255 for i in range(settings.EMBEDDING_DIMENSIONS)]


def make_pdf(pages: list[str]) -> bytes:
    """Build a PDF in memory; an empty string makes a page with no text (like a scan)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text)
    return doc.tobytes()


class FakeGemini:
    def __init__(self):
        self.embed_calls = 0
        self.error: Exception | None = None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        if self.error:
            raise self.error
        return [fake_vector(t) for t in texts]


@pytest.fixture
def fake_gemini(monkeypatch) -> FakeGemini:
    fake = FakeGemini()
    monkeypatch.setattr(GeminiService, "embed_documents", fake.embed_documents)
    return fake


@pytest.fixture
def test_collection(monkeypatch):
    """Point Qdrant at a throwaway collection so tests never touch real documents."""
    monkeypatch.setattr(settings, "QDRANT_COLLECTION", TEST_COLLECTION)
    qdrant_service.qdrant_client.delete_collection(TEST_COLLECTION)
    qdrant_service.ensure_collection()
    yield TEST_COLLECTION
    qdrant_service.qdrant_client.delete_collection(TEST_COLLECTION)


@pytest.fixture
async def client(fake_gemini, test_collection):
    """HTTP client for the app. Deletes every document the test created afterwards."""
    # Pooled connections belong to the event loop that opened them; each test gets a new loop.
    await engine.dispose()
    created: list[str] = []

    async def capture(response: httpx.Response):
        if (
            response.request.method == "POST"
            and response.request.url.path == "/documents"
        ):
            await response.aread()
            document = response.json().get("document") or {}
            if "id" in document:
                created.append(document["id"])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        event_hooks={"response": [capture]},
    ) as http:
        yield http

    async with SessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(created)))
        await db.commit()
    await engine.dispose()
