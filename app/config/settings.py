from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="./.env",
        env_ignore_empty=True,
        extra="ignore",
    )

    APP_ENV: str = "development"

    DB_HOST: str = "mysql"
    DB_PORT: int = 3306
    DB_DATABASE: str = "rag_db"
    DB_USERNAME: str = "rag_user"
    DB_PASSWORD: str = "change_me"

    @computed_field
    def DATABASE_URI(self) -> str:
        return f"mysql+asyncmy://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_DATABASE}"

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"

    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 768
    # Gemini accepts up to 100 texts per embed request.
    EMBEDDING_BATCH_SIZE: int = Field(default=100, ge=1, le=100)

    # Chat answers and follow-up rewrites use any OpenAI-compatible API; Groq's free tier
    # by default. Gemini's free tier allows only ~20 answer requests a day.
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "openai/gpt-oss-120b"
    # Follow-up rewriting is a simple task, so it uses a smaller model. Rate limits are
    # counted per model, so rewrites don't use up the answer model's requests.
    REWRITE_MODEL: str = "openai/gpt-oss-20b"

    # Chunking, in characters. ~800 chars (~200 tokens) keeps one idea per chunk for precise
    # citations; 120 chars (15%) of overlap stops a sentence from being split across chunks.
    CHUNK_SIZE: int = Field(default=800, ge=100)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)

    # Number of chunks retrieved per question.
    TOP_K: int = Field(default=5, ge=1, le=50)
    # Relevance floor: chunks scoring below this never reach the LLM, and if none pass the
    # assistant says it lacks information without an LLM call. Kept low on purpose: correct
    # chunks can score as low as ~0.63 (e.g. a side detail in a multi-topic chunk) while
    # unrelated ones reach ~0.63 too, so no single cut-off separates them. Clearly unrelated
    # questions (~0.47-0.52) are dropped here; the LLM's `answerable` flag decides the rest.
    MIN_SCORE: float = Field(default=0.55, ge=0.0, le=1.0)

    MAX_UPLOAD_MB: int = Field(default=10, ge=1)

    # Latest N chat messages (user + assistant combined) sent as conversation context.
    HISTORY_LIMIT: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _check_chunk_overlap(self):
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
