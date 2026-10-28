import json

import httpx
import pymupdf
import pytest
from sqlalchemy import delete

from app.config.database import SessionLocal, engine
from app.config.settings import settings
from app.main import app
from app.models import Conversation, Document
from app.services import qdrant_service
from app.services.gemini_service import GeminiService
from app.services.llm_service import LLMService
from app.services.prompt_service import StandaloneQuery

TEST_COLLECTION = "documents_test"


# Topic words the fake embedding understands. Texts sharing a topic score ~0.99 cosine
# similarity; texts with no topic in common score ~0, so tests control what is retrieved.
TOPICS = ["leave", "travel", "remote", "salary", "parking"]


def fake_vector(text: str) -> list[float]:
    """Deterministic 768-dim vector, so tests need no API key and cost nothing."""
    words = text.lower().replace(".", " ").replace("?", " ").split()
    vector = [0.0] * settings.EMBEDDING_DIMENSIONS
    for i, topic in enumerate(TOPICS):
        vector[i] = float(words.count(topic))
    # Small constant so a text with no topic words is not an all-zero vector.
    vector[-1] = 0.01
    return vector


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
        self.generate_calls = 0
        self.error: Exception | None = None
        # What generate() returns; tests override it to simulate other LLM behaviour.
        self.reply = {
            "answer": "Fake answer.",
            "answerable": True,
            "used_sources": ["S1"],
        }
        self.last_prompt: dict | None = None
        # Rewrite result; empty means "search the original question".
        self.rewrite_to = ""
        self.rewrite_calls = 0
        self.rewrite_error: Exception | None = None
        self.last_rewrite_prompt: str | None = None
        self.last_rewrite_model: str | None = None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        if self.error:
            raise self.error
        return [fake_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        if self.error:
            raise self.error
        return fake_vector(text)

    async def generate(
        self, system_instruction, messages, response_schema=None, model=None, **_
    ):
        if response_schema is StandaloneQuery:
            # Follow-up rewrite: counted apart from answer calls.
            self.rewrite_calls += 1
            self.last_rewrite_prompt = messages[-1]["content"]
            self.last_rewrite_model = model
            if self.rewrite_error:
                raise self.rewrite_error
            return json.dumps({"query": self.rewrite_to or ""})

        self.generate_calls += 1
        self.last_prompt = {"system": system_instruction, "messages": messages}
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


@pytest.fixture
def fake_gemini(monkeypatch) -> FakeGemini:
    fake = FakeGemini()
    for name in ("embed_documents", "embed_query"):
        monkeypatch.setattr(GeminiService, name, getattr(fake, name))
    monkeypatch.setattr(LLMService, "generate", fake.generate)
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
    """HTTP client for the app. Deletes every document and conversation the test created."""
    # Pooled connections belong to the event loop that opened them; each test gets a new loop.
    await engine.dispose()
    documents: list[str] = []
    conversations: list[str] = []

    async def capture(response: httpx.Response):
        if response.request.method != "POST" or response.status_code >= 500:
            return
        path = response.request.url.path
        if path == "/documents":
            await response.aread()
            document = response.json().get("document") or {}
            if "id" in document:
                documents.append(document["id"])
        elif path == "/conversations":
            await response.aread()
            conversations.append(response.json()["id"])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        event_hooks={"response": [capture]},
    ) as http:
        yield http

    async with SessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(documents)))
        # Messages go with their conversation (ON DELETE CASCADE).
        await db.execute(delete(Conversation).where(Conversation.id.in_(conversations)))
        await db.commit()
    await engine.dispose()
