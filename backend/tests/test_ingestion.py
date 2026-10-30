from app.clients import vector_store_client
from app.clients.embedding_client import EmbeddingClientError
from app.config.settings import settings
from tests.conftest import make_pdf

POLICY_TEXT = " ".join(
    f"Employees receive 15 days of paid leave, rule {i}." for i in range(60)
)


async def upload(client, filename: str, data: bytes, content_type: str = "text/plain"):
    return await client.post(
        "/documents", files={"file": (filename, data, content_type)}
    )


async def test_upload_pdf_indexes_chunks_with_pages(client, fake_gemini):
    data = make_pdf([POLICY_TEXT, "Remote work is allowed 2 days a week."])

    response = await upload(client, "policy.pdf", data, "application/pdf")

    assert response.status_code == 201, response.text
    body = response.json()["data"]
    document = body["document"]
    version = document["versions"][0]
    assert body["duplicate"] is False
    assert document["status"] == "indexed"
    assert document["active_version"] == 1
    assert version["page_count"] == 2
    assert version["chunk_count"] > 1
    assert vector_store_client.count_chunks(document["id"], 1) == version["chunk_count"]

    # Every stored chunk carries the metadata citations need.
    points, _ = vector_store_client.qdrant_client.scroll(
        settings.QDRANT_COLLECTION, limit=100, with_payload=True
    )
    assert {p.payload["page"] for p in points} == {1, 2}
    assert all(p.payload["filename"] == "policy.pdf" for p in points)
    assert all(p.payload["document_id"] == document["id"] for p in points)


async def test_get_and_list_documents(client, fake_gemini):
    created = (await upload(client, "leave.txt", b"15 days of leave.")).json()["data"]
    document_id = created["document"]["id"]

    one = await client.get(f"/documents/{document_id}")
    listed = await client.get("/documents")

    assert one.status_code == 200
    assert one.json()["data"]["filename"] == "leave.txt"
    assert document_id in [d["id"] for d in listed.json()["data"]]


async def test_get_unknown_document_is_404(client):
    response = await client.get("/documents/999999")

    assert response.status_code == 404


async def test_duplicate_upload_returns_existing_document(client, fake_gemini):
    data = b"Employees receive 15 days of paid leave per year."
    first = (await upload(client, "leave.txt", data)).json()["data"]

    second = await upload(client, "renamed.txt", data)

    assert second.status_code == 200
    assert second.json()["data"]["duplicate"] is True
    assert second.json()["data"]["document"]["id"] == first["document"]["id"]
    # Nothing was embedded or stored a second time.
    assert fake_gemini.embed_calls == 1
    assert vector_store_client.count_chunks(first["document"]["id"]) == 1


async def test_failed_file_can_be_uploaded_again(client, fake_gemini):
    data = b"Remote work is allowed 2 days a week."
    fake_gemini.error = EmbeddingClientError(
        "Gemini embedding request failed (503 UNAVAILABLE)"
    )
    failed = await upload(client, "remote.txt", data)
    fake_gemini.error = None

    retry = await upload(client, "remote.txt", data)

    assert failed.status_code == 422
    assert retry.status_code == 201
    assert retry.json()["data"]["duplicate"] is False


async def test_scanned_pdf_fails_with_reason(client, fake_gemini):
    response = await upload(client, "scan.pdf", make_pdf(["", ""]), "application/pdf")

    assert response.status_code == 422
    document = response.json()["data"]["document"]
    assert document["status"] == "failed"
    assert document["active_version"] is None
    assert "OCR is not supported" in document["versions"][0]["error"]
    assert fake_gemini.embed_calls == 0


async def test_gemini_failure_marks_failed_and_stores_nothing(client, fake_gemini):
    fake_gemini.error = EmbeddingClientError(
        "Gemini embedding request failed (429 RESOURCE_EXHAUSTED)"
    )

    response = await upload(client, "notes.txt", b"Some text to index.")

    assert response.status_code == 422
    document = response.json()["data"]["document"]
    assert document["status"] == "failed"
    assert "429" in document["versions"][0]["error"]
    assert vector_store_client.count_chunks(document["id"]) == 0


async def test_unsupported_type_is_rejected(client, fake_gemini):
    before = len((await client.get("/documents")).json()["data"])

    response = await upload(client, "report.docx", b"PK fake docx")

    assert response.status_code == 400
    assert "Unsupported file type '.docx'" in response.json()["detail"]
    # Rejected before any document row was created.
    assert len((await client.get("/documents")).json()["data"]) == before


async def test_fake_pdf_is_rejected(client):
    response = await upload(client, "fake.pdf", b"hello", "application/pdf")

    assert response.status_code == 400
    assert response.json()["detail"] == "The file is not a valid PDF"


async def test_empty_file_is_rejected(client):
    response = await upload(client, "empty.txt", b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "The file is empty"


async def test_file_over_size_limit_is_rejected(client, fake_gemini, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)

    response = await upload(client, "big.txt", b"a" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert fake_gemini.embed_calls == 0


async def test_client_path_is_stripped_from_filename(client, fake_gemini):
    response = await upload(client, "C:\\Users\\me\\leave.txt", b"15 days of leave.")

    assert response.json()["data"]["document"]["filename"] == "leave.txt"
