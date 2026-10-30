"""Shared test setup.

The app runs for real against MySQL and a throwaway Qdrant collection. Only the two paid
outside APIs are replaced: embeddings (Gemini) and the chat LLM (Groq). That keeps the
tests free, needing no API keys, and giving the same result on every run.
"""

import json
import re
import uuid
from pathlib import Path

import httpx
import pymupdf
import pytest
from sqlalchemy import delete

from app.clients import vector_store_client
from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClient
from app.config.database import SessionLocal, engine
from app.config.settings import settings
from app.main import app
from app.models import Conversation, Document
from app.services.prompt_service import LLMAnswer

TEST_COLLECTION = "documents_test"

_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    _ROOT / "samples" if (_ROOT / "samples").is_dir() else _ROOT.parent / "samples"
)

# Each topic word owns one dimension of the fake vector. Texts that share a topic score
# close to 1.0 cosine similarity, texts with no topic in common score close to 0, so a
# test decides what gets retrieved just by the words it uses.
TOPICS = [
    "annual", "leave", "sick", "parental", "holiday", "remote",
    "travel", "meal", "salary", "parking", "diabetes", "sugar",
]  # fmt: skip


def fake_vector(text: str) -> list[float]:
    words = re.findall(r"[a-z]+", text.lower())
    vector = [0.0] * settings.EMBEDDING_DIMENSIONS
    for i, topic in enumerate(TOPICS):
        vector[i] = float(sum(word.startswith(topic) for word in words))
    # Never an all-zero vector: Qdrant cannot score one.
    vector[-1] = 0.01
    return vector


def make_pdf(*pages: str) -> bytes:
    """A real PDF with one text page per argument."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    return document.tobytes()


def sample(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


def unique(data: bytes) -> bytes:
    """The same text plus a random last line, so the upload is never a duplicate.

    Tests share MySQL with the dev app, where the same sample may already be indexed.
    """
    return data + f"\n\nTest run {uuid.uuid4().hex}\n".encode()


class FakeProviders:
    """Stands in for Gemini embeddings and the Groq chat LLM, and records every call."""

    def __init__(self):
        # Embeddings
        self.embed_batches: list[list[str]] = []
        self.embed_error: Exception | None = None
        # Chat LLM
        self.reply: dict = {
            "answer": "Fake answer.",
            "answerable": True,
            "used_sources": ["S1"],
        }
        self.answer_prompts: list[list[dict]] = []
        self.rewrite_prompts: list[list[dict]] = []
        self.rewrite: str | None = None
        self.raw_reply: str | None = None
        self.error: Exception | None = None
        self.rewrite_error: Exception | None = None

    async def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if self.embed_error:
            raise self.embed_error
        self.embed_batches.append(list(texts))
        return [fake_vector(text) for text in texts]

    async def generate(
        self,
        system_instruction,
        messages,
        response_schema=None,
        temperature=0.0,
        model=None,
    ) -> str:
        if response_schema is LLMAnswer:
            self.answer_prompts.append(messages)
            if self.error:
                raise self.error
            return self.raw_reply or json.dumps(self.reply)

        self.rewrite_prompts.append(messages)
        if self.rewrite_error:
            raise self.rewrite_error
        question = re.search(
            r"<question>\n(.*)\n</question>", messages[-1]["content"], re.S
        )
        return json.dumps({"query": self.rewrite or question.group(1)})

    @property
    def last_prompt(self) -> str:
        """The newest user turn sent to the answer model: the context and the question."""
        return self.answer_prompts[-1][-1]["content"]


@pytest.fixture
def fake_providers(monkeypatch) -> FakeProviders:
    fake = FakeProviders()
    monkeypatch.setattr(EmbeddingClient, "_embed", fake.embed)
    monkeypatch.setattr(LLMClient, "generate", fake.generate)
    return fake


def _reset_collection() -> None:
    # A fresh Qdrant container can be slow to answer its first requests.
    for attempt in range(3):
        try:
            vector_store_client.qdrant_client.delete_collection(TEST_COLLECTION)
            vector_store_client.ensure_collection()
            return
        except Exception:
            if attempt == 2:
                raise


@pytest.fixture
def test_collection(monkeypatch):
    """Point Qdrant at a throwaway collection so tests never touch real documents."""
    monkeypatch.setattr(settings, "QDRANT_COLLECTION", TEST_COLLECTION)
    _reset_collection()
    yield TEST_COLLECTION
    vector_store_client.qdrant_client.delete_collection(TEST_COLLECTION)


@pytest.fixture
async def client(fake_providers, test_collection):
    """HTTP client for the app. Deletes every document and conversation the test made."""
    # Pooled connections belong to the event loop that opened them; each test has its own.
    await engine.dispose()
    documents: list[int] = []
    conversations: list[int] = []

    async def remember_created(response: httpx.Response):
        if response.request.method != "POST" or response.status_code >= 500:
            return
        await response.aread()
        data = response.json().get("data") or {}
        path = response.request.url.path
        if path == "/documents" and "document" in data:
            documents.append(data["document"]["id"])
        elif path == "/conversations":
            conversations.append(data["id"])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        event_hooks={"response": [remember_created]},
    ) as http:
        yield http

    async with SessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(documents)))
        await db.execute(delete(Conversation).where(Conversation.id.in_(conversations)))
        await db.commit()
    await engine.dispose()


# Helpers used by many test files.


async def upload(client, filename: str, data: bytes, content_type="text/plain"):
    return await client.post(
        "/documents", files={"file": (filename, data, content_type)}
    )


async def upload_ok(client, filename: str, data: bytes, content_type="text/plain"):
    response = await upload(client, filename, data, content_type)
    assert response.status_code == 201, response.text
    return response.json()["data"]["document"]


async def replace(client, document_id: int, filename: str, data: bytes):
    return await client.put(
        f"/documents/{document_id}", files={"file": (filename, data, "text/plain")}
    )


async def new_conversation(client) -> int:
    response = await client.post("/conversations")
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


async def ask(client, conversation_id: int, question: str, **extra):
    return await client.post(
        f"/chat/{conversation_id}", json={"question": question, **extra}
    )


async def ask_ok(client, conversation_id: int, question: str, **extra) -> dict:
    response = await ask(client, conversation_id, question, **extra)
    assert response.status_code == 200, response.text
    return response.json()["data"]
