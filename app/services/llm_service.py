import asyncio
import json
import logging
import random
from typing import Literal, TypedDict

import httpx
from pydantic import BaseModel

from app.config.settings import settings

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 60.0

# Rate limits (429) and provider hiccups (5xx) are usually brief, so retry before failing.
# Waits grow roughly 2s, 4s (plus jitter), or follow the provider's retry-after header.
_RETRY_ATTEMPTS = 3
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_INITIAL_DELAY_S = 2.0
# A longer requested wait means a daily quota or similar, so fail at once instead of hanging.
_MAX_RETRY_WAIT_S = 20.0


class LLMServiceError(Exception):
    """Any failure talking to the chat LLM. Messages never include the API key."""


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LLMService:
    """Chat completions over any OpenAI-compatible API (Groq by default, also OpenRouter
    or a local Ollama). Embeddings stay on Gemini, see GeminiService."""

    _client: httpx.AsyncClient | None = None

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        if not settings.LLM_API_KEY:
            raise LLMServiceError("LLM_API_KEY is not set")
        if cls._client is None:
            cls._client = httpx.AsyncClient(
                base_url=settings.LLM_BASE_URL.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                timeout=_REQUEST_TIMEOUT_S,
            )
        return cls._client

    @classmethod
    async def generate(
        cls,
        system_instruction: str,
        messages: list[ChatMessage],
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        """Generate a reply. System instructions are passed separately from user/document content.

        With `response_schema` (a Pydantic model) the reply is a JSON object. JSON mode works on
        every OpenAI-compatible provider but does not enforce the shape, so the schema is also
        described in the system instruction and callers validate the reply.
        `model` defaults to LLM_MODEL.
        """
        system = system_instruction
        body: dict = {
            "model": model or settings.LLM_MODEL,
            "temperature": temperature,
        }
        if response_schema is not None:
            schema = json.dumps(response_schema.model_json_schema())
            system += (
                "\n\nReply with only a JSON object that matches this JSON schema:\n"
                f"{schema}"
            )
            body["response_format"] = {"type": "json_object"}
        body["messages"] = [{"role": "system", "content": system}, *messages]

        data = await cls._post("/chat/completions", body)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMServiceError("LLM returned an unexpected response") from e
        if not text:
            raise LLMServiceError("LLM returned an empty response")
        return text

    @classmethod
    async def _post(cls, path: str, body: dict) -> dict:
        """POST with retries; every failure becomes LLMServiceError."""
        client = cls._get_client()
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                response = await client.post(path, json=body)
            except httpx.HTTPError as e:
                logger.warning("LLM request failed: %s", type(e).__name__)
                if attempt == _RETRY_ATTEMPTS:
                    raise LLMServiceError("LLM request failed") from e
                await asyncio.sleep(cls._backoff(attempt))
                continue

            if response.is_success:
                return response.json()

            # Only the status is surfaced; the provider message stays in server logs.
            logger.warning(
                "LLM request failed: %s %s",
                response.status_code,
                response.text[:500],
            )
            wait = cls._retry_wait(response, attempt)
            if (
                response.status_code not in _RETRY_STATUS_CODES
                or attempt == _RETRY_ATTEMPTS
                or wait > _MAX_RETRY_WAIT_S
            ):
                raise LLMServiceError(f"LLM request failed ({response.status_code})")
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return _INITIAL_DELAY_S * 2 ** (attempt - 1) + random.uniform(0, 1)

    @classmethod
    def _retry_wait(cls, response: httpx.Response, attempt: int) -> float:
        try:
            return float(response.headers["retry-after"])
        except (KeyError, ValueError):
            return cls._backoff(attempt)
