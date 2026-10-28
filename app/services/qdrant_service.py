import threading
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.config.settings import settings
from app.schemas.chunk import ChunkPayload

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


# Fixed namespace so the same (document_id, version, chunk_index) always maps to the same point ID.
_CHUNK_ID_NAMESPACE = uuid.UUID("6f1c2a52-3c1e-4c55-9a43-1d0b6c2f8e17")
_UPSERT_BATCH_SIZE = 100


def make_chunk_id(document_id: str, version: int, chunk_index: int) -> str:
    """Deterministic point ID: re-indexing the same version overwrites instead of duplicating."""
    return str(
        uuid.uuid5(_CHUNK_ID_NAMESPACE, f"{document_id}:{version}:{chunk_index}")
    )


def _document_filter(document_id: str, version: int | None = None) -> models.Filter:
    conditions = [
        models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id)
        )
    ]
    if version is not None:
        conditions.append(
            models.FieldCondition(key="version", match=models.MatchValue(value=version))
        )
    return models.Filter(must=conditions)


def upsert_chunks(payloads: list[ChunkPayload], vectors: list[list[float]]) -> None:
    """Store chunks with their vectors. Returns only after Qdrant has applied every write."""
    if len(payloads) != len(vectors):
        raise ValueError(f"Got {len(payloads)} payloads but {len(vectors)} vectors")

    points = [
        models.PointStruct(id=p.chunk_id, vector=v, payload=p.model_dump())
        for p, v in zip(payloads, vectors, strict=True)
    ]
    for start in range(0, len(points), _UPSERT_BATCH_SIZE):
        qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points[start : start + _UPSERT_BATCH_SIZE],
            wait=True,
        )


def count_chunks(document_id: str, version: int | None = None) -> int:
    """Number of stored chunks for a document, or for one version of it."""
    return qdrant_client.count(
        collection_name=settings.QDRANT_COLLECTION,
        count_filter=_document_filter(document_id, version),
        exact=True,
    ).count


def delete_version(document_id: str, version: int) -> None:
    """Remove one version's chunks, e.g. the old version after an update succeeds."""
    qdrant_client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=_document_filter(document_id, version)
        ),
        wait=True,
    )


def delete_document(document_id: str) -> None:
    """Remove every chunk of a document, across all versions."""
    qdrant_client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=_document_filter(document_id)),
        wait=True,
    )


@dataclass(frozen=True)
class SearchHit:
    payload: ChunkPayload
    score: float


def search(
    vector: list[float],
    active_versions: list[tuple[str, int]],
    top_k: int,
) -> list[SearchHit]:
    """Return the `top_k` most similar chunks, best first.

    Only chunks whose (document_id, version) is in `active_versions` can match, so retired
    versions and deleted documents are never returned, even if their points still exist.
    No score cut-off is applied here; the caller decides what counts as evidence.
    """
    if not active_versions:
        return []

    # Match any one of the live (document_id, version) pairs.
    version_filter = models.Filter(
        should=[
            _document_filter(document_id, version)
            for document_id, version in active_versions
        ]
    )
    response = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        query_filter=version_filter,
        limit=top_k,
        with_payload=True,
    )
    return [
        SearchHit(payload=ChunkPayload.model_validate(point.payload), score=point.score)
        for point in response.points
    ]
