import logging
from typing import Literal, TypedDict

from google import genai
from google.genai import errors, types

from app.config.settings import settings

logger = logging.getLogger(__name__)

# The SDK expects milliseconds.
_REQUEST_TIMEOUT_MS = 60_000

# Gemini calls the assistant role "model".
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiServiceError(Exception):
    """Any failure talking to Gemini. Messages never include the API key."""


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GeminiService:
    _client: genai.Client | None = None

    @classmethod
    def _get_client(cls) -> genai.Client:
        if not settings.GEMINI_API_KEY:
            raise GeminiServiceError("GEMINI_API_KEY is not set")
        if cls._client is None:
            cls._client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
            )
        return cls._client

    @classmethod
    async def embed_documents(cls, texts: list[str]) -> list[list[float]]:
        """Embed document chunks in batches. Output order matches input order."""
        vectors: list[list[float]] = []
        batch_size = settings.EMBEDDING_BATCH_SIZE
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await cls._embed(batch, task_type="RETRIEVAL_DOCUMENT"))
        return vectors

    @classmethod
    async def embed_query(cls, text: str) -> list[float]:
        """Embed a user question. Uses the query task type so it matches document embeddings."""
        return (await cls._embed([text], task_type="RETRIEVAL_QUERY"))[0]

    @classmethod
    async def generate(
        cls,
        system_instruction: str,
        messages: list[ChatMessage],
        response_schema: type | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a reply. System instructions are passed separately from user/document content.

        With `response_schema` (e.g. a Pydantic model) the reply is JSON matching that schema.
        """
        contents = [
            types.Content(role=_ROLE_MAP[m["role"]], parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json" if response_schema else None,
            response_schema=response_schema,
        )

        response = await cls._call(
            "generate",
            cls._get_client().aio.models.generate_content(
                model=settings.LLM_MODEL, contents=contents, config=config
            ),
        )
        if not response.text:
            raise GeminiServiceError("Gemini returned an empty response")
        return response.text

    @classmethod
    async def _embed(cls, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        response = await cls._call(
            "embedding",
            cls._get_client().aio.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.EMBEDDING_DIMENSIONS,
                ),
            ),
        )

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise GeminiServiceError(
                f"Gemini returned {len(embeddings)} embeddings for {len(texts)} texts"
            )
        vectors = [e.values or [] for e in embeddings]
        if any(len(v) != settings.EMBEDDING_DIMENSIONS for v in vectors):
            raise GeminiServiceError(
                f"Gemini returned embeddings with unexpected dimensions (expected {settings.EMBEDDING_DIMENSIONS})"
            )
        return vectors

    @classmethod
    async def _call(cls, operation: str, request):
        """Await a Gemini request and convert every failure into GeminiServiceError."""
        try:
            return await request
        except errors.APIError as e:
            # Only the status is surfaced; the provider message stays in server logs.
            logger.warning("Gemini %s request failed: %s %s", operation, e.code, e.status)
            raise GeminiServiceError(
                f"Gemini {operation} request failed ({e.code} {e.status})"
            ) from e
        except Exception as e:
            logger.exception("Gemini %s request failed", operation)
            raise GeminiServiceError(f"Gemini {operation} request failed") from e
