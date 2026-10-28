from pydantic import BaseModel


class ChunkPayload(BaseModel):
    """Metadata stored with each vector in Qdrant.

    `document_id` + `version` let retrieval ignore retired versions and let delete
    remove every chunk of a document.
    """

    document_id: str
    version: int
    chunk_id: str
    chunk_index: int
    filename: str
    # Null for sources without pages (TXT, URL).
    page: int | None = None
    content_hash: str
    source_url: str | None = None
    text: str
