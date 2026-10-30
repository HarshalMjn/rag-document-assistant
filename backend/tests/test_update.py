from datetime import datetime

from sqlalchemy import update

from app.clients import vector_store_client
from app.clients.embedding_client import EmbeddingClient, EmbeddingClientError
from app.config.database import SessionLocal
from app.config.settings import settings
from app.models import Document
from app.services.retrieval_service import RetrievalService
from tests.conftest import make_pdf
from tests.test_chat import ask, new_conversation, upload

V1_TEXT = b"Employees receive 15 days of paid leave per year."
V2_TEXT = b"Employees receive 20 days of paid leave per year."


async def put(
    client, document_id: int, filename: str, data: bytes, content_type="text/plain"
):
    return await client.put(
        f"/documents/{document_id}", files={"file": (filename, data, content_type)}
    )


async def test_update_indexes_a_new_version_and_removes_the_old(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)

    response = await put(client, document["id"], "leave.txt", V2_TEXT)

    assert response.status_code == 200, response.text
    body = response.json()["data"]
    updated = body["document"]
    assert body["duplicate"] is False
    assert updated["id"] == document["id"]
    assert updated["active_version"] == 2
    assert updated["status"] == "indexed"
    assert [(v["version"], v["status"]) for v in updated["versions"]] == [
        (1, "indexed"),
        (2, "indexed"),
    ]
    assert vector_store_client.count_chunks(document["id"], 1) == 0
    assert vector_store_client.count_chunks(document["id"], 2) == 1


async def test_chat_after_update_uses_only_the_new_content(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    await put(client, document["id"], "leave.txt", V2_TEXT)
    conversation_id = await new_conversation(client)

    body = (await ask(client, conversation_id, "How many leave days?")).json()["data"]

    assert {t["version"] for t in body["retrieval_trace"]} == {2}
    prompt = fake_gemini.last_prompt["messages"][-1]["content"]
    assert "20 days" in prompt
    assert "15 days" not in prompt


async def test_old_version_stays_live_while_the_new_one_is_indexed(
    client, fake_gemini, monkeypatch
):
    document = await upload(client, "leave.txt", V1_TEXT)
    seen_during_indexing = {}
    embed = fake_gemini.embed_documents

    async def checking_embed(texts):
        async with SessionLocal() as db:
            seen_during_indexing["live"] = await RetrievalService.active_versions(db)
        seen_during_indexing["v1_chunks"] = vector_store_client.count_chunks(
            document["id"], 1
        )
        return await embed(texts)

    monkeypatch.setattr(EmbeddingClient, "embed_documents", checking_embed)

    await put(client, document["id"], "leave.txt", V2_TEXT)

    assert (document["id"], 1) in seen_during_indexing["live"]
    assert seen_during_indexing["v1_chunks"] == 1


async def test_unchanged_content_is_not_reindexed(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    embeds_before = fake_gemini.embed_calls

    response = await put(client, document["id"], "leave.txt", V1_TEXT)

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["duplicate"] is True
    assert body["document"]["active_version"] == 1
    assert len(body["document"]["versions"]) == 1
    assert fake_gemini.embed_calls == embeds_before


async def test_failed_update_keeps_the_old_version_live(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    fake_gemini.error = EmbeddingClientError("Gemini embedding request failed (503)")

    response = await put(client, document["id"], "new-leave.txt", V2_TEXT)

    assert response.status_code == 422
    updated = response.json()["data"]["document"]
    assert updated["status"] == "indexed"
    assert updated["active_version"] == 1
    # The name only changes once a new version is indexed.
    assert updated["filename"] == "leave.txt"
    failed = updated["versions"][-1]
    assert (failed["version"], failed["status"]) == (2, "failed")
    assert "503" in failed["error"]
    assert vector_store_client.count_chunks(document["id"], 1) == 1
    assert vector_store_client.count_chunks(document["id"], 2) == 0


async def test_version_after_a_failed_update_gets_the_next_number(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    fake_gemini.error = EmbeddingClientError("Gemini embedding request failed (503)")
    await put(client, document["id"], "leave.txt", V2_TEXT)
    fake_gemini.error = None

    response = await put(client, document["id"], "leave.txt", V2_TEXT)

    assert response.status_code == 200
    updated = response.json()["data"]["document"]
    assert updated["active_version"] == 3
    assert [v["status"] for v in updated["versions"]] == [
        "indexed",
        "failed",
        "indexed",
    ]
    assert vector_store_client.count_chunks(document["id"]) == 1


async def test_update_can_change_name_and_type(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    pdf = make_pdf([V2_TEXT.decode()])

    response = await put(client, document["id"], "leave.pdf", pdf, "application/pdf")

    assert response.status_code == 200, response.text
    assert response.json()["data"]["document"]["filename"] == "leave.pdf"
    points, _ = vector_store_client.qdrant_client.scroll(
        settings.QDRANT_COLLECTION, limit=10, with_payload=True
    )
    assert [p.payload["filename"] for p in points] == ["leave.pdf"]
    assert points[0].payload["page"] == 1


async def test_update_of_unknown_document_is_404(client):
    response = await put(client, 999999, "leave.txt", V2_TEXT)

    assert response.status_code == 404


async def test_update_of_deleted_document_is_404(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    async with SessionLocal() as db:
        await db.execute(
            update(Document)
            .where(Document.id == document["id"])
            .values(deleted_at=datetime.now())
        )
        await db.commit()

    response = await put(client, document["id"], "leave.txt", V2_TEXT)

    assert response.status_code == 404


async def test_update_validates_the_file_like_upload(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)

    empty = await put(client, document["id"], "leave.txt", b"")
    wrong_type = await put(client, document["id"], "leave.docx", V2_TEXT)

    assert empty.status_code == 400
    assert wrong_type.status_code == 400
