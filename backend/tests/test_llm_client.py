import json

import httpx
import pytest

from app.clients import llm_client
from app.clients.llm_client import LLMClient, LLMClientError
from app.config.settings import settings
from app.services.prompt_service import StandaloneQuery


def ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def provider(monkeypatch):
    """Route LLMClient to a fake provider. Append responses to `replies`; requests land in `seen`."""
    replies: list[httpx.Response] = []
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return replies.pop(0)

    monkeypatch.setattr(settings, "CHAT_LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        LLMClient,
        "_client",
        httpx.AsyncClient(
            base_url="http://llm.test", transport=httpx.MockTransport(handle)
        ),
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm_client.asyncio, "sleep", no_sleep)
    return replies, seen


async def test_schema_requests_json_mode_and_describes_shape(provider):
    replies, seen = provider
    replies.append(ok('{"query": "sick leave days"}'))

    raw = await LLMClient.generate(
        "Rewrite.", [{"role": "user", "content": "q"}], response_schema=StandaloneQuery
    )

    assert StandaloneQuery.model_validate_json(raw).query == "sick leave days"
    body = seen[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == settings.CHAT_ANSWER_MODEL
    assert body["messages"][0]["role"] == "system"
    assert '"query"' in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "q"}


async def test_model_override(provider):
    replies, seen = provider
    replies.append(ok("hi"))

    await LLMClient.generate("s", [], model="small-model")

    assert seen[0]["model"] == "small-model"
    assert "response_format" not in seen[0]


async def test_short_rate_limit_is_retried(provider):
    replies, seen = provider
    replies += [httpx.Response(429, headers={"retry-after": "2"}), ok("done")]

    assert await LLMClient.generate("s", []) == "done"
    assert len(seen) == 2


async def test_long_rate_limit_fails_at_once(provider):
    """A daily quota asks for hours of waiting; retrying would only delay the error."""
    replies, seen = provider
    replies.append(httpx.Response(429, headers={"retry-after": "3600"}))

    with pytest.raises(LLMClientError, match=r"\(429\)"):
        await LLMClient.generate("s", [])
    assert len(seen) == 1


async def test_bad_key_is_not_retried(provider):
    replies, seen = provider
    replies.append(httpx.Response(401, json={"error": "invalid key"}))

    with pytest.raises(LLMClientError, match=r"\(401\)"):
        await LLMClient.generate("s", [])
    assert len(seen) == 1


async def test_missing_key_fails_without_a_request(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_LLM_API_KEY", "")
    monkeypatch.setattr(LLMClient, "_client", None)

    with pytest.raises(LLMClientError, match="CHAT_LLM_API_KEY is not set"):
        await LLMClient.generate("s", [])
