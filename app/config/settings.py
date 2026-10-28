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
    LLM_MODEL: str = "gemini-2.5-flash"
    # Gemini accepts up to 100 texts per embed request.
    EMBEDDING_BATCH_SIZE: int = Field(default=100, ge=1, le=100)

    # Chunking, in characters. ~800 chars (~200 tokens) keeps one idea per chunk for precise
    # citations; 120 chars (15%) of overlap stops a sentence from being split across chunks.
    CHUNK_SIZE: int = Field(default=800, ge=100)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)

    # Number of chunks retrieved per question.
    TOP_K: int = Field(default=5, ge=1, le=50)

    MAX_UPLOAD_MB: int = Field(default=10, ge=1)

    # Latest N chat messages (user + assistant combined) sent as conversation context.
    HISTORY_LIMIT: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _check_chunk_overlap(self):
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
