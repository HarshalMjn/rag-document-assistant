import json

import httpx
import pytest

from app.config.settings import settings
from app.services import llm_service
from app.services.llm_service import LLMService, LLMServiceError
from app.services.prompt_service import StandaloneQuery


def ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def provider(monkeypatch):
    """Route LLMService to a fake provider. Append responses to `replies`; requests land in `seen`."""
    replies: list[httpx.Response] = []
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return replies.pop(0)

    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        LLMService,
        "_client",
        httpx.AsyncClient(
            base_url="http://llm.test", transport=httpx.MockTransport(handle)
        ),
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    return replies, seen


async def test_schema_requests_json_mode_and_describes_shape(provider):
    replies, seen = provider
    replies.append(ok('{"query": "sick leave days"}'))

    raw = await LLMService.generate(
        "Rewrite.", [{"role": "user", "content": "q"}], response_schema=StandaloneQuery
    )

    assert StandaloneQuery.model_validate_json(raw).query == "sick leave days"
    body = seen[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == settings.LLM_MODEL
    assert body["messages"][0]["role"] == "system"
    assert '"query"' in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "q"}


async def test_model_override(provider):
    replies, seen = provider
    replies.append(ok("hi"))

    await LLMService.generate("s", [], model="small-model")

    assert seen[0]["model"] == "small-model"
    assert "response_format" not in seen[0]


async def test_short_rate_limit_is_retried(provider):
    replies, seen = provider
    replies += [httpx.Response(429, headers={"retry-after": "2"}), ok("done")]

    assert await LLMService.generate("s", []) == "done"
    assert len(seen) == 2


async def test_long_rate_limit_fails_at_once(provider):
    """A daily quota asks for hours of waiting; retrying would only delay the error."""
    replies, seen = provider
    replies.append(httpx.Response(429, headers={"retry-after": "3600"}))

    with pytest.raises(LLMServiceError, match=r"\(429\)"):
        await LLMService.generate("s", [])
    assert len(seen) == 1


async def test_bad_key_is_not_retried(provider):
    replies, seen = provider
    replies.append(httpx.Response(401, json={"error": "invalid key"}))

    with pytest.raises(LLMServiceError, match=r"\(401\)"):
        await LLMService.generate("s", [])
    assert len(seen) == 1


async def test_missing_key_fails_without_a_request(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(LLMService, "_client", None)

    with pytest.raises(LLMServiceError, match="LLM_API_KEY is not set"):
        await LLMService.generate("s", [])
