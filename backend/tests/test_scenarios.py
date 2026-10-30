"""Required test scenarios from the assignment (section 6), and where each is covered.

| Scenario              | Tests                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| Basic retrieval       | test_chat.py::test_answer_cites_the_retrieved_chunk                   |
| Unsupported question  | test_chat.py::test_no_relevant_chunk_answers_insufficient_without_llm |
|                       | test_chat.py::test_llm_saying_not_answerable_gives_insufficient       |
|                       | test_chat.py::test_invented_source_is_dropped                         |
| Follow-up context     | test_history.py (all), test_chat.py::test_follow_up_shows_rewritten_* |
| History boundary      | test_history_boundary_uses_only_the_latest_10_messages (here)         |
|                       | test_history.py::test_window_drops_an_answer_whose_question_was_cut_off |
| Document update       | test_leave_policy_update_from_15_to_20_days (here), test_update.py    |
| Document deletion     | test_delete.py                                                        |
| Duplicate upload      | test_ingestion.py::test_duplicate_upload_returns_existing_document    |
|                       | test_update.py::test_unchanged_content_is_not_reindexed               |
| Bad input             | test_ingestion.py: empty, fake PDF, scanned PDF, unsupported, too big |
| Multiple sources      | test_answer_cites_both_sources (here)                                 |
| Failure handling      | test_chat.py::test_llm_failure_returns_502_and_stores_nothing         |
|                       | test_embedding_failure_during_chat_returns_502 (here)                 |
|                       | test_ingestion.py::test_gemini_failure_marks_failed_and_stores_nothing |
|                       | test_update.py::test_failed_update_keeps_the_old_version_live         |
"""

from app.clients.embedding_client import EmbeddingClientError
from tests.test_chat import ask, new_conversation, stored_messages, upload
from tests.test_update import put

LEAVE_15 = b"Leave policy: annual leave is 15 days per year."
LEAVE_20 = b"Leave policy: annual leave is 20 days per year."
TRAVEL_TEXT = b"Travel meals are reimbursed up to $50 per day."


async def test_leave_policy_update_from_15_to_20_days(client, fake_gemini):
    """The exact procedure from the assignment: ask, update the same document, ask again."""
    policy = await upload(client, "policy.txt", LEAVE_15)
    conversation_id = await new_conversation(client)

    before = (
        await ask(client, conversation_id, "How many days of annual leave?")
    ).json()["data"]
    assert "15 days" in fake_gemini.last_prompt["messages"][-1]["content"]
    assert {t["version"] for t in before["retrieval_trace"]} == {1}

    response = await put(client, policy["id"], "policy.txt", LEAVE_20)
    assert response.status_code == 200
    assert response.json()["data"]["document"]["id"] == policy["id"]

    after = (
        await ask(client, conversation_id, "How many days of annual leave?")
    ).json()["data"]
    prompt = fake_gemini.last_prompt["messages"][-1]["content"]
    # Only the new version reaches the LLM; no stale chunk is retrieved.
    assert "20 days" in prompt
    assert "15 days" not in prompt
    assert {t["version"] for t in after["retrieval_trace"]} == {2}
    assert [c["document_id"] for c in after["citations"]] == [policy["id"]]


async def test_history_boundary_uses_only_the_latest_10_messages(client, fake_gemini):
    """12 messages are stored; the 7th question sees only the latest 10 (5 full pairs)."""
    await upload(client, "leave.txt", LEAVE_15)
    conversation_id = await new_conversation(client)
    for turn in range(1, 7):
        fake_gemini.reply = {**fake_gemini.reply, "answer": f"Answer {turn}."}
        await ask(client, conversation_id, f"Leave question {turn}?")
    assert len(await stored_messages(conversation_id)) == 12

    await ask(client, conversation_id, "Leave question 7?")

    history = fake_gemini.last_prompt["messages"][:-1]
    assert len(history) == 10
    # Turn 1 is the oldest and falls outside the window; turns 2-6 remain, oldest first.
    assert history[0] == {"role": "user", "content": "Leave question 2?"}
    assert history[-1] == {"role": "assistant", "content": "Answer 6."}
    assert "Leave question 1?" not in str(history)


async def test_answer_cites_both_sources(client, fake_gemini):
    leave = await upload(client, "leave.txt", LEAVE_15)
    travel = await upload(client, "travel.txt", TRAVEL_TEXT)
    conversation_id = await new_conversation(client)
    fake_gemini.reply = {
        "answer": "Leave is 15 days, and travel meals are covered up to $50 a day.",
        "answerable": True,
        "used_sources": ["S1", "S2"],
    }

    body = (
        await ask(client, conversation_id, "What are the leave and travel rules?")
    ).json()["data"]

    # Evidence from both documents reached the LLM, and both are cited.
    prompt = fake_gemini.last_prompt["messages"][-1]["content"]
    assert "leave.txt" in prompt and "travel.txt" in prompt
    assert {c["document_id"] for c in body["citations"]} == {leave["id"], travel["id"]}
    assert body["insufficient_evidence"] is False


async def test_embedding_failure_during_chat_returns_502(client, fake_gemini):
    await upload(client, "leave.txt", LEAVE_15)
    conversation_id = await new_conversation(client)
    fake_gemini.error = EmbeddingClientError("Gemini embedding request failed (503)")

    response = await ask(client, conversation_id, "How many leave days?")

    assert response.status_code == 502
    assert "503" in response.json()["detail"]
    # No false success: neither the question nor an answer is stored.
    assert await stored_messages(conversation_id) == []
