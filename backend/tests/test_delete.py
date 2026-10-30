from app.clients import vector_store_client
from tests.test_chat import ask, new_conversation, upload
from tests.test_update import V1_TEXT, V2_TEXT, put

TRAVEL_TEXT = b"Travel meals are reimbursed up to $50 per day."


async def test_delete_removes_the_document_and_its_chunks(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)

    response = await client.delete(f"/documents/{document['id']}")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "id": document["id"],
        "deleted": True,
        "chunks_removed": 1,
    }
    assert vector_store_client.count_chunks(document["id"]) == 0
    assert (await client.get(f"/documents/{document['id']}")).status_code == 404
    listed = (await client.get("/documents")).json()["data"]
    assert document["id"] not in [d["id"] for d in listed]


async def test_delete_removes_chunks_of_every_version(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    await put(client, document["id"], "leave.txt", V2_TEXT)

    response = await client.delete(f"/documents/{document['id']}")

    assert response.json()["data"]["chunks_removed"] == 1
    assert vector_store_client.count_chunks(document["id"]) == 0


async def test_chat_never_cites_a_deleted_document(client, fake_gemini):
    leave = await upload(client, "leave.txt", V1_TEXT)
    travel = await upload(client, "travel.txt", TRAVEL_TEXT)
    await client.delete(f"/documents/{leave['id']}")
    conversation_id = await new_conversation(client)

    body = (await ask(client, conversation_id, "How many leave days?")).json()["data"]

    assert leave["id"] not in {t["document_id"] for t in body["retrieval_trace"]}
    assert leave["id"] not in {c["document_id"] for c in body["citations"]}
    # The other document is still searchable.
    assert travel["id"] in {t["document_id"] for t in body["retrieval_trace"]}


async def test_document_stays_deleted_if_qdrant_cleanup_fails(
    client, fake_gemini, monkeypatch
):
    document = await upload(client, "leave.txt", V1_TEXT)

    def broken_delete(document_id):
        raise ConnectionError("Qdrant is down")

    monkeypatch.setattr(vector_store_client, "delete_document", broken_delete)
    conversation_id = await new_conversation(client)

    response = await client.delete(f"/documents/{document['id']}")
    body = (await ask(client, conversation_id, "How many leave days?")).json()["data"]

    assert response.status_code == 200
    assert response.json()["data"]["chunks_removed"] == 0
    # The chunks are still in Qdrant, but MySQL says the document is gone.
    assert vector_store_client.count_chunks(document["id"]) == 1
    assert body["retrieval_trace"] == []
    assert body["insufficient_evidence"] is True


async def test_delete_twice_is_404(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    await client.delete(f"/documents/{document['id']}")

    response = await client.delete(f"/documents/{document['id']}")

    assert response.status_code == 404


async def test_delete_unknown_document_is_404(client):
    response = await client.delete("/documents/999999")

    assert response.status_code == 404


async def test_deleted_document_cannot_be_updated(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    await client.delete(f"/documents/{document['id']}")

    response = await put(client, document["id"], "leave.txt", V2_TEXT)

    assert response.status_code == 404


async def test_same_file_can_be_uploaded_again_after_delete(client, fake_gemini):
    document = await upload(client, "leave.txt", V1_TEXT)
    await client.delete(f"/documents/{document['id']}")

    response = await client.post(
        "/documents", files={"file": ("leave.txt", V1_TEXT, "text/plain")}
    )

    # Not a duplicate of the deleted document: it is indexed again as a new one.
    assert response.status_code == 201
    body = response.json()["data"]
    assert body["duplicate"] is False
    assert body["document"]["id"] != document["id"]
