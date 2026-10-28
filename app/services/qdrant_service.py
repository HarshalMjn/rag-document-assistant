import threading

from qdrant_client import QdrantClient, models

from app.config.settings import settings

_client_lock = threading.Lock()


def _make_qdrant_client() -> QdrantClient:
    kwargs = {
        "host": settings.QDRANT_HOST,
        "port": settings.QDRANT_PORT,
    }
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
        # qdrant-client defaults to HTTPS when api_key is set; self-hosted Qdrant only serves HTTP.
        kwargs["https"] = False
    return QdrantClient(**kwargs)


class _LazyQdrantClient:
    def __init__(self):
        self._client = None

    def _get(self) -> QdrantClient:
        if self._client is None:
            with _client_lock:
                if self._client is None:
                    self._client = _make_qdrant_client()
        return self._client

    def __getattr__(self, name):
        return getattr(self._get(), name)


qdrant_client = _LazyQdrantClient()


def ensure_collection() -> None:
    """Create the documents collection and its payload indexes if missing."""
    name = settings.QDRANT_COLLECTION
    if qdrant_client.collection_exists(name):
        return

    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=settings.EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )
    # Retrieval filters on these fields to exclude stale versions and deleted documents.
    qdrant_client.create_payload_index(
        collection_name=name,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    qdrant_client.create_payload_index(
        collection_name=name,
        field_name="version",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
