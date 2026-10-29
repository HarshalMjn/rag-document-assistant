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
    DB_USERNAME: str
    DB_PASSWORD: str

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
    EMBEDDING_BATCH_SIZE: int = Field(default=100, ge=1, le=100)

    CHAT_LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    CHAT_LLM_API_KEY: str = ""
    CHAT_ANSWER_MODEL: str = "openai/gpt-oss-120b"
    CHAT_REWRITE_MODEL: str = "openai/gpt-oss-20b"

    CHUNK_SIZE: int = Field(default=800, ge=100)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)

    TOP_K: int = Field(default=5, ge=1, le=50)
    MIN_SCORE: float = Field(default=0.55, ge=0.0, le=1.0)

    MAX_UPLOAD_MB: int = Field(default=10, ge=1)

    HISTORY_LIMIT: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _check_chunk_overlap(self):
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
