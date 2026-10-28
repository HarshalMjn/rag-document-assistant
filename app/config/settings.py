from pydantic import computed_field
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


settings = Settings()
