"""The 10 test scenarios from section 6 of the assignment, one group of tests each.

Runs with fake embeddings and a fake LLM (see conftest.py), so it needs no API keys:
    docker compose exec fastapi_app pytest tests/test_scenarios.py
The fake LLM says whatever a test tells it to, so these tests check our side of the
pipeline: what is retrieved, what reaches the prompt, what is cited and stored.
Real Gemini/Groq checks are in tests/live/ (pytest -m live).
"""

import pytest
from conftest import (
    ask,
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
from app.clients.llm_client import LLMClientError
from app.services.prompt_service import INSUFFICIENT_ANSWER

ANNUAL_QUESTION = "How many annual leave days do I get?"


async def history(client, conversation_id: int) -> list[dict]:
    response = await client.get(f"/chat/{conversation_id}/history")
    assert response.status_code == 200, response.text
    return response.json()["data"]["messages"]


# 1. Basic retrieval


async def test_basic_retrieval(client, fake_providers):
    document = await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )
    fake_providers.reply = {
        "answer": "Full-time employees get 15 days of paid annual leave.",
        "answerable": True,
        "used_sources": ["S1"],
    }

    data = await ask_ok(client, await new_conversation(client), ANNUAL_QUESTION)

    assert "15 days" in fake_providers.last_prompt
    assert data["answer"] == fake_providers.reply["answer"]
    assert data["insufficient_evidence"] is False
    [citation] = data["citations"]
    assert citation["document_id"] == document["id"]
    assert citation["filename"] == "leave_policy_v1.txt"
    assert "15 days" in citation["text"]


# 2. Unsupported question


async def test_unsupported_question_says_insufficient_evidence(client, fake_providers):
    await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )

    data = await ask_ok(
        client, await new_conversation(client), "How much does a parking permit cost?"
    )

    # No chunk is relevant, so the LLM is never asked and cannot invent anything.
    assert fake_providers.answer_prompts == []
    assert data["answer"] == INSUFFICIENT_ANSWER
    assert data["insufficient_evidence"] is True
    assert data["citations"] == []


async def test_unsupported_question_drops_invented_citation(client, fake_providers):
    await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )
    # The model claims a source that was never in the prompt.
    fake_providers.reply = {
        "answer": "Parking costs 40 euros.",
        "answerable": True,
        "used_sources": ["S9"],
    }

    data = await ask_ok(client, await new_conversation(client), ANNUAL_QUESTION)

    assert data["answer"] == INSUFFICIENT_ANSWER
    assert data["insufficient_evidence"] is True
    assert data["citations"] == []


# 3. Follow-up context


async def test_follow_up_uses_recent_context(client, fake_providers):
    document = await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )
    conversation_id = await new_conversation(client)
    await ask_ok(client, conversation_id, ANNUAL_QUESTION)

    # "them" only makes sense with the first question; the fake rewrite resolves it.
    fake_providers.rewrite = "annual leave carry over"
    data = await ask_ok(client, conversation_id, "Can I carry them over?")

    rewrite_prompt = fake_providers.rewrite_prompts[-1][-1]["content"]
    assert f"user: {ANNUAL_QUESTION}" in rewrite_prompt
    assert "assistant: Fake answer." in rewrite_prompt
    assert data["search_query"] == "annual leave carry over"
    assert data["history_used"] == 2
    # The answer model also sees the earlier turn, before the new question.
    assert fake_providers.answer_prompts[-1][0]["content"] == ANNUAL_QUESTION
    assert [c["document_id"] for c in data["citations"]] == [document["id"]]


# 4. History boundary


async def test_history_uses_only_latest_10_messages(client, fake_providers):
    await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )
    conversation_id = await new_conversation(client)
    # 6 turns = 12 stored messages (each turn is one user + one assistant message).
    for n in range(1, 7):
        await ask_ok(client, conversation_id, f"Annual leave question case {n}?")

    data = await ask_ok(client, conversation_id, "Annual leave question case 7?")

    # The latest 10 messages: turns 2 to 6. Turn 1 has dropped out of the window.
    assert data["history_used"] == 10
    prompt = fake_providers.answer_prompts[-1]
    assert len(prompt) == 11  # 10 history messages + the new question
    assert prompt[0] == {"role": "user", "content": "Annual leave question case 2?"}
    assert not any("case 1?" in message["content"] for message in prompt)
    # Nothing is deleted: the full history is still stored.
    assert len(await history(client, conversation_id)) == 14


# 5. Document update


async def test_document_update_replaces_old_chunks(client, fake_providers):
    document = await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )

    response = await replace(
        client,
        document["id"],
        "leave_policy_v2.txt",
        unique(sample("leave_policy_v2.txt")),
    )

    assert response.status_code == 200, response.text
    updated = response.json()["data"]
    assert updated["duplicate"] is False
    assert updated["document"]["id"] == document["id"]
    assert updated["document"]["active_version"] == 2
    assert vector_store_client.count_chunks(document["id"], version=1) == 0

    data = await ask_ok(client, await new_conversation(client), ANNUAL_QUESTION)

    assert "20 days" in fake_providers.last_prompt
    assert "15 days" not in fake_providers.last_prompt
    assert data["retrieval_trace"]
    assert {item["version"] for item in data["retrieval_trace"]} == {2}
    assert data["citations"][0]["filename"] == "leave_policy_v2.txt"


# 6. Document deletion


async def test_deleted_document_is_not_retrieved(client, fake_providers):
    document = await upload_ok(
        client,
        "parking.txt",
        unique(
            b"Staff parking permits cost 40 euros per month. Parking is free at weekends."
        ),
    )
    conversation_id = await new_conversation(client)
    question = "How much does a parking permit cost?"
    before = await ask_ok(client, conversation_id, question)
    assert [c["document_id"] for c in before["citations"]] == [document["id"]]

    response = await client.delete(f"/documents/{document['id']}")

    assert response.status_code == 200, response.text
    assert response.json()["data"]["chunks_removed"] >= 1
    assert vector_store_client.count_chunks(document["id"]) == 0

    calls_before = len(fake_providers.answer_prompts)
    after = await ask_ok(client, conversation_id, question)

    assert len(fake_providers.answer_prompts) == calls_before  # nothing left to send
    assert after["answer"] == INSUFFICIENT_ANSWER
    assert after["citations"] == []
    assert all(i["document_id"] != document["id"] for i in after["retrieval_trace"])
    assert (await client.get(f"/documents/{document['id']}")).status_code == 404


# 7. Duplicate upload


async def test_duplicate_upload_returns_existing_document(client, fake_providers):
    data = unique(b"Remote work is allowed for up to 2 days per week.")
    first = await upload(client, "remote.txt", data)
    embed_calls = len(fake_providers.embed_batches)

    second = await upload(client, "remote_copy.txt", data)

    # Documented behaviour: same bytes -> 200, the existing document, nothing re-embedded.
    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert second.json()["data"]["duplicate"] is True
    assert (
        second.json()["data"]["document"]["id"]
        == first.json()["data"]["document"]["id"]
    )
    assert len(fake_providers.embed_batches) == embed_calls


async def test_update_with_same_content_is_a_no_op(client, fake_providers):
    data = unique(b"Remote work is allowed for up to 2 days per week.")
    document = await upload_ok(client, "remote.txt", data)

    response = await replace(client, document["id"], "remote.txt", data)

    assert response.status_code == 200, response.text
    assert response.json()["data"]["duplicate"] is True
    assert response.json()["data"]["document"]["active_version"] == 1


# 8. Bad input


@pytest.mark.parametrize(
    ("filename", "content_type", "detail"),
    [
        ("empty.txt", "text/plain", "The file is empty"),
        ("not_a_pdf.pdf", "application/pdf", "not a valid PDF"),
        ("unsupported_type.csv", "text/csv", "Unsupported file type"),
    ],
)
async def test_bad_input_is_rejected(client, filename, content_type, detail):
    response = await upload(
        client, filename, sample(f"bad_inputs/{filename}"), content_type
    )

    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]
    await _assert_app_still_works(client)


async def test_scanned_pdf_fails_with_reason(client, fake_providers):
    response = await upload(
        client,
        "scanned_page.pdf",
        sample("bad_inputs/scanned_page.pdf"),
        "application/pdf",
    )

    # The PDF is valid, so a document row exists, but indexing fails with a clear reason.
    assert response.status_code == 422, response.text
    document = response.json()["data"]["document"]
    assert document["active_version"] is None
    assert document["versions"][-1]["status"] == "failed"
    assert "No extractable text" in document["versions"][-1]["error"]
    assert fake_providers.embed_batches == []
    await _assert_app_still_works(client)


async def _assert_app_still_works(client):
    assert (await client.get("/health")).status_code == 200
    await upload_ok(client, "ok.txt", unique(b"Annual leave is 20 days."))


# 9. Multiple sources


async def test_answer_cites_both_sources(client, fake_providers):
    leave = await upload_ok(
        client,
        "leave.txt",
        unique(b"Annual leave rule: every employee gets 20 days of annual leave."),
    )
    travel = await upload_ok(
        client,
        "travel.txt",
        unique(b"Travel rule: the meal allowance when travelling is 30 euros per day."),
    )
    fake_providers.reply = {
        "answer": "You get 20 days of annual leave and a 30 euro daily meal allowance.",
        "answerable": True,
        "used_sources": ["S1", "S2"],
    }

    data = await ask_ok(
        client,
        await new_conversation(client),
        "How much annual leave and travel meal allowance do I get?",
    )

    assert "20 days of annual leave" in fake_providers.last_prompt
    assert "meal allowance" in fake_providers.last_prompt
    assert {c["document_id"] for c in data["citations"]} == {leave["id"], travel["id"]}
    assert {c["filename"] for c in data["citations"]} == {"leave.txt", "travel.txt"}


# 10. Failure handling


async def test_embedding_failure_during_chat(client, fake_providers):
    await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )
    conversation_id = await new_conversation(client)
    fake_providers.embed_error = EmbeddingClientError("Gemini quota exceeded")

    response = await ask(client, conversation_id, ANNUAL_QUESTION)

    assert response.status_code == 502, response.text
    assert "embedding service is unavailable" in response.json()["detail"]
    assert await history(client, conversation_id) == []  # no half-saved turn


@pytest.mark.parametrize(
    ("error", "raw_reply"),
    [
        (LLMClientError("Groq timed out"), None),
        (None, "this is not JSON"),
    ],
    ids=["provider-error", "invalid-reply"],
)
async def test_llm_failure_during_chat(client, fake_providers, error, raw_reply):
    await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )
    conversation_id = await new_conversation(client)
    fake_providers.error = error
    fake_providers.raw_reply = raw_reply

    response = await ask(client, conversation_id, ANNUAL_QUESTION)

    assert response.status_code == 502, response.text
    assert "language model is unavailable" in response.json()["detail"]
    assert await history(client, conversation_id) == []


async def test_embedding_failure_during_upload(client, fake_providers):
    fake_providers.embed_error = EmbeddingClientError("Gemini quota exceeded")

    response = await upload(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )

    assert response.status_code == 422, response.text
    document = response.json()["data"]["document"]
    assert document["active_version"] is None
    assert document["versions"][-1]["status"] == "failed"
    assert "Gemini quota exceeded" in document["versions"][-1]["error"]
    assert vector_store_client.count_chunks(document["id"]) == 0
