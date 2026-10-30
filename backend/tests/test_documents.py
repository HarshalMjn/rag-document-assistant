"""Requirements 3D and 3E: updating and deleting a particular document.

Checks what test_scenarios.py does not: a failed update keeps the old version live,
version numbering, timestamps, retired chunks never retrieved even if Qdrant still holds
them, and the delete edge cases. Runs with fake providers, so it needs no API keys.
"""

import asyncio

from conftest import (
    ask_ok,
    new_conversation,
    replace,
    sample,
    unique,
    upload,
    upload_ok,
)

from app.clients import vector_store_client
from app.clients.embedding_client import EmbeddingClientError

ANNUAL_QUESTION = "How many annual leave days do I get?"
MISSING_DOCUMENT = 999_999_999


async def upload_v1(client) -> dict:
    return await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )


async def ask_annual(client) -> dict:
    return await ask_ok(client, await new_conversation(client), ANNUAL_QUESTION)


def qdrant_fails(*args, **kwargs):
    raise RuntimeError("Qdrant is unavailable")


# Update


async def test_failed_update_keeps_the_previous_version_live(client, fake_providers):
    document = await upload_v1(client)
    fake_providers.embed_error = EmbeddingClientError("Gemini quota exceeded")

    response = await replace(
        client,
        document["id"],
        "leave_policy_v2.txt",
        unique(sample("leave_policy_v2.txt")),
    )

    # Not reported as success, and the failure is visible on the new version.
    assert response.status_code == 422, response.text
    updated = response.json()["data"]["document"]
    assert updated["active_version"] == 1
    assert updated["status"] == "indexed"
    failed = updated["versions"][-1]
    assert (failed["version"], failed["status"]) == (2, "failed")
    assert "Gemini quota exceeded" in failed["error"]
    assert vector_store_client.count_chunks(document["id"], version=2) == 0

    # Version 1 still answers questions.
    fake_providers.embed_error = None
    data = await ask_annual(client)
    assert "15 days" in fake_providers.last_prompt
    assert {item["version"] for item in data["retrieval_trace"]} == {1}


async def test_version_numbers_skip_a_failed_version(client, fake_providers):
    document = await upload_v1(client)
    v2 = unique(sample("leave_policy_v2.txt"))
    fake_providers.embed_error = EmbeddingClientError("Gemini quota exceeded")
    await replace(client, document["id"], "leave_policy_v2.txt", v2)

    fake_providers.embed_error = None
    response = await replace(client, document["id"], "leave_policy_v2.txt", v2)

    assert response.status_code == 200, response.text
    updated = response.json()["data"]["document"]
    assert updated["active_version"] == 3
    assert updated["filename"] == "leave_policy_v2.txt"
    assert [(v["version"], v["status"]) for v in updated["versions"][1:]] == [
        (2, "failed"),
        (3, "indexed"),
    ]
    assert vector_store_client.count_chunks(document["id"], version=1) == 0


async def test_update_changes_updated_at(client):
    document = await upload_v1(client)
    # MySQL timestamps have 1-second precision.
    await asyncio.sleep(1.1)

    response = await replace(
        client,
        document["id"],
        "leave_policy_v2.txt",
        unique(sample("leave_policy_v2.txt")),
    )

    updated = response.json()["data"]["document"]
    assert updated["created_at"] == document["created_at"]
    assert updated["updated_at"] > document["updated_at"]
    assert updated["versions"][-1]["created_at"] > document["versions"][0]["created_at"]


async def test_invalid_replacement_is_rejected_before_any_change(client):
    document = await upload_v1(client)

    response = await replace(client, document["id"], "empty.txt", b"")

    assert response.status_code == 400, response.text
    current = (await client.get(f"/documents/{document['id']}")).json()["data"]
    assert len(current["versions"]) == 1
    assert current["active_version"] == 1


async def test_retired_chunks_are_never_retrieved(client, fake_providers, monkeypatch):
    document = await upload_v1(client)
    # The old version's chunks cannot be removed, so they stay in Qdrant.
    monkeypatch.setattr(vector_store_client, "delete_version", qdrant_fails)

    response = await replace(
        client,
        document["id"],
        "leave_policy_v2.txt",
        unique(sample("leave_policy_v2.txt")),
    )

    assert response.status_code == 200, response.text
    v1_chunks = vector_store_client.count_chunks(document["id"], version=1)
    assert v1_chunks > 0
    # Retrieval only searches active versions, so the leftovers never come back.
    data = await ask_annual(client)
    assert {item["version"] for item in data["retrieval_trace"]} == {2}
    assert "15 days" not in fake_providers.last_prompt

    # Deleting the document removes every version's chunks, leftovers included.
    v2_chunks = vector_store_client.count_chunks(document["id"], version=2)
    deleted = await client.delete(f"/documents/{document['id']}")
    assert deleted.json()["data"]["chunks_removed"] == v1_chunks + v2_chunks
    assert vector_store_client.count_chunks(document["id"]) == 0


# Delete


async def test_deleted_document_leaves_the_list(client):
    document = await upload_v1(client)

    response = await client.delete(f"/documents/{document['id']}")

    assert response.status_code == 200, response.text
    assert response.json()["data"] == {
        "id": document["id"],
        "deleted": True,
        "chunks_removed": document["versions"][0]["chunk_count"],
    }
    listed = (await client.get("/documents")).json()["data"]
    assert document["id"] not in [d["id"] for d in listed]


async def test_upload_after_delete_creates_a_new_document(client):
    data = unique(sample("leave_policy_v1.txt"))
    first = await upload_ok(client, "leave_policy_v1.txt", data)
    await client.delete(f"/documents/{first['id']}")

    response = await upload(client, "leave_policy_v1.txt", data)

    # A deleted document no longer counts as a duplicate.
    assert response.status_code == 201, response.text
    assert response.json()["data"]["duplicate"] is False
    assert response.json()["data"]["document"]["id"] != first["id"]


async def test_delete_succeeds_even_if_qdrant_cleanup_fails(
    client, fake_providers, monkeypatch
):
    document = await upload_v1(client)
    monkeypatch.setattr(vector_store_client, "delete_document", qdrant_fails)

    response = await client.delete(f"/documents/{document['id']}")

    # MySQL is the source of truth: the document is gone even with chunks left behind.
    assert response.status_code == 200, response.text
    assert response.json()["data"]["chunks_removed"] == 0
    assert vector_store_client.count_chunks(document["id"]) > 0
    data = await ask_annual(client)
    assert all(i["document_id"] != document["id"] for i in data["retrieval_trace"])
    assert data["citations"] == []


async def test_unknown_or_deleted_document_returns_404(client):
    document = await upload_v1(client)
    await client.delete(f"/documents/{document['id']}")
    data = unique(sample("leave_policy_v2.txt"))

    for document_id in (MISSING_DOCUMENT, document["id"]):
        responses = [
            await client.get(f"/documents/{document_id}"),
            await replace(client, document_id, "leave_policy_v2.txt", data),
            await client.delete(f"/documents/{document_id}"),
        ]
        for response in responses:
            assert response.status_code == 404, response.text
            assert response.json()["detail"] == "Document not found"
