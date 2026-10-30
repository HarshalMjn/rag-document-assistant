"""Requirement 3C: conversation history (latest 10 messages).

Checks what test_scenarios.py does not: the incomplete pair at the window boundary, a
configurable limit, separate conversations, the history endpoint, the rewrite fallback
and that history never replaces fresh retrieval. Runs with fake providers, no API keys.
"""

import pytest
from conftest import (
    ask,
    ask_ok,
    new_conversation,
    replace,
    sample,
    unique,
    upload_ok,
)

from app.clients.llm_client import LLMClientError
from app.config.settings import settings

ANNUAL_QUESTION = "How many annual leave days do I get?"
MISSING_CONVERSATION = 999_999_999


async def history(client, conversation_id: int) -> list[dict]:
    response = await client.get(f"/chat/{conversation_id}/history")
    assert response.status_code == 200, response.text
    return response.json()["data"]["messages"]


@pytest.fixture
async def leave_policy(client) -> dict:
    return await upload_ok(
        client, "leave_policy_v1.txt", unique(sample("leave_policy_v1.txt"))
    )


async def ask_turns(client, conversation_id: int, count: int) -> None:
    for n in range(1, count + 1):
        await ask_ok(client, conversation_id, f"Annual leave question case {n}?")


# Window


async def test_incomplete_pair_at_the_boundary_is_dropped(
    client, fake_providers, leave_policy, monkeypatch
):
    monkeypatch.setattr(settings, "HISTORY_LIMIT", 3)
    conversation_id = await new_conversation(client)
    await ask_turns(client, conversation_id, 2)

    data = await ask_ok(client, conversation_id, "Annual leave question case 3?")

    # The latest 3 messages are [answer 1, question 2, answer 2]. Answer 1 has lost its
    # question, so it is dropped and the window starts with a user message.
    assert data["history_used"] == 2
    prompt = fake_providers.answer_prompts[-1]
    assert prompt[0] == {"role": "user", "content": "Annual leave question case 2?"}
    assert prompt[1]["role"] == "assistant"
    assert len(prompt) == 3  # 2 history messages + the new question


@pytest.mark.parametrize(("limit", "expected"), [(4, 4), (0, 0)])
async def test_history_limit_is_configurable(
    client, fake_providers, leave_policy, monkeypatch, limit, expected
):
    monkeypatch.setattr(settings, "HISTORY_LIMIT", limit)
    conversation_id = await new_conversation(client)
    await ask_turns(client, conversation_id, 3)

    data = await ask_ok(client, conversation_id, "Annual leave question case 4?")

    assert data["history_used"] == expected
    assert len(fake_providers.answer_prompts[-1]) == expected + 1


async def test_first_question_skips_the_rewrite(client, fake_providers, leave_policy):
    data = await ask_ok(client, await new_conversation(client), ANNUAL_QUESTION)

    assert fake_providers.rewrite_prompts == []
    assert data["search_query"] == ANNUAL_QUESTION
    assert data["history_used"] == 0


# Separate conversations


async def test_conversations_do_not_share_history(client, fake_providers, leave_policy):
    first = await new_conversation(client)
    second = await new_conversation(client)
    await ask_ok(client, first, "Annual leave question from the first chat?")

    data = await ask_ok(client, second, "Annual leave question from the second chat?")

    assert data["history_used"] == 0
    assert fake_providers.rewrite_prompts == []  # the second chat has no history yet
    prompt = fake_providers.answer_prompts[-1]
    assert len(prompt) == 1
    assert "first chat" not in prompt[0]["content"]
    assert [m["content"] for m in await history(client, second)] == [
        "Annual leave question from the second chat?",
        "Fake answer.",
    ]
    assert len(await history(client, first)) == 2


# History endpoint


async def test_history_endpoint_returns_messages_in_order(
    client, fake_providers, leave_policy
):
    conversation_id = await new_conversation(client)
    await ask_turns(client, conversation_id, 2)

    messages = await history(client, conversation_id)

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages[::2]] == [
        "Annual leave question case 1?",
        "Annual leave question case 2?",
    ]
    assert [m["id"] for m in messages] == sorted(m["id"] for m in messages)
    # Answers keep their citations, so old turns can still show their sources.
    assert messages[0]["citations"] is None
    assert messages[1]["citations"][0]["document_id"] == leave_policy["id"]


async def test_conversation_list_shows_title_and_message_count(client, leave_policy):
    conversation_id = await new_conversation(client)
    await ask_turns(client, conversation_id, 2)

    response = await client.get("/conversations")

    assert response.status_code == 200, response.text
    [summary] = [c for c in response.json()["data"] if c["id"] == conversation_id]
    assert summary["title"] == "Annual leave question case 1?"
    assert summary["message_count"] == 4


async def test_unknown_conversation_returns_404(client):
    chat = await ask(client, MISSING_CONVERSATION, ANNUAL_QUESTION)
    messages = await client.get(f"/chat/{MISSING_CONVERSATION}/history")

    for response in (chat, messages):
        assert response.status_code == 404, response.text
        assert response.json()["detail"] == "Conversation not found"


# Follow-ups


async def test_failed_rewrite_falls_back_to_the_original_question(
    client, fake_providers, leave_policy
):
    conversation_id = await new_conversation(client)
    await ask_ok(client, conversation_id, ANNUAL_QUESTION)
    fake_providers.rewrite_error = LLMClientError("Groq timed out")

    data = await ask_ok(client, conversation_id, "How much annual leave carries over?")

    # The rewrite is only a search helper, so its failure does not fail the chat.
    assert len(fake_providers.rewrite_prompts) == 1
    assert data["search_query"] == "How much annual leave carries over?"
    assert data["history_used"] == 2
    assert data["citations"]


async def test_follow_up_retrieves_fresh_evidence(client, fake_providers, leave_policy):
    conversation_id = await new_conversation(client)
    fake_providers.reply = {
        "answer": "You get 15 days of annual leave.",
        "answerable": True,
        "used_sources": ["S1"],
    }
    await ask_ok(client, conversation_id, ANNUAL_QUESTION)
    await replace(
        client,
        leave_policy["id"],
        "leave_policy_v2.txt",
        unique(sample("leave_policy_v2.txt")),
    )

    fake_providers.rewrite = "annual leave days"
    await ask_ok(client, conversation_id, "Is that still true?")

    # The old answer is in the history, but the facts come from the current document.
    prompt = fake_providers.answer_prompts[-1]
    assert prompt[1] == {
        "role": "assistant",
        "content": "You get 15 days of annual leave.",
    }
    assert "20 days" in fake_providers.last_prompt
    assert "15 days" not in fake_providers.last_prompt
