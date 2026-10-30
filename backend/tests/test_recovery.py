from sqlalchemy import update

from app.clients import vector_store_client
from app.config.database import SessionLocal
from app.models import DocumentVersion
from app.models.document import Document, DocumentStatus
from app.schemas.chunk import ChunkPayload
from app.services.ingestion_service import INTERRUPTED_ERROR, IngestionService
from tests.conftest import fake_vector

LEAVE_TEXT = b"Employees receive 15 days of paid leave per year."


async def upload(client, filename: str, data: bytes) -> dict:
    response = await client.post(
        "/documents", files={"file": (filename, data, "text/plain")}
    )
    return response.json()["data"]["document"]


async def recover() -> int:
    async with SessionLocal() as db:
        return await IngestionService.recover_interrupted(db)


async def test_interrupted_first_upload_is_marked_failed(client, fake_gemini):
    document = await upload(client, "leave.txt", LEAVE_TEXT)
    # Simulate a crash mid-indexing: the version never left `processing`.
    async with SessionLocal() as db:
        await db.execute(
            update(Document)
            .where(Document.id == document["id"])
            .values(active_version_id=None)
        )
        await db.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document["id"])
            .values(status=DocumentStatus.PROCESSING)
        )
        await db.commit()

    assert await recover() == 1

    body = (await client.get(f"/documents/{document['id']}")).json()["data"]
    assert body["status"] == "failed"
    assert body["active_version"] is None
    assert body["versions"][0]["error"] == INTERRUPTED_ERROR
    assert vector_store_client.count_chunks(document["id"]) == 0


async def test_interrupted_update_keeps_previous_version_live(client, fake_gemini):
    document = await upload(client, "leave.txt", LEAVE_TEXT)
    # Simulate a crash while version 2 was being indexed: some of its chunks are stored.
    async with SessionLocal() as db:
        db.add(
            DocumentVersion(
                document_id=document["id"],
                version=2,
                content_hash="v2",
                status=DocumentStatus.PROCESSING,
                filename="leave.txt",
            )
        )
        await db.commit()
    partial = ChunkPayload(
        document_id=document["id"],
        version=2,
        chunk_id=vector_store_client.make_chunk_id(document["id"], 2, 0),
        chunk_index=0,
        filename="leave.txt",
        content_hash="v2",
        text="Employees receive 20 days of paid leave per year.",
    )
    vector_store_client.upsert_chunks([partial], [fake_vector(partial.text)])

    assert await recover() == 1

    body = (await client.get(f"/documents/{document['id']}")).json()["data"]
    assert body["status"] == "indexed"
    assert body["active_version"] == 1
    assert [(v["version"], v["status"]) for v in body["versions"]] == [
        (1, "indexed"),
        (2, "failed"),
    ]
    assert vector_store_client.count_chunks(document["id"], 2) == 0
    assert vector_store_client.count_chunks(document["id"], 1) == 1


async def test_recovery_ignores_finished_versions(client, fake_gemini):
    await upload(client, "leave.txt", LEAVE_TEXT)

    assert await recover() == 0
