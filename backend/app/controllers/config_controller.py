from app.config.settings import settings


class ConfigController:
    @classmethod
    def config(cls) -> dict:
        """Non-secret settings the UI shows. Keys, hosts and passwords are never included."""
        return {
            "data": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "top_k": settings.TOP_K,
                "min_score": settings.MIN_SCORE,
                "history_limit": settings.HISTORY_LIMIT,
                "max_upload_mb": settings.MAX_UPLOAD_MB,
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.CHAT_ANSWER_MODEL,
                "rewrite_model": settings.CHAT_REWRITE_MODEL,
            }
        }
