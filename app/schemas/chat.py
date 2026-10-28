from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.message import MessageRole


class ChatRequest(BaseModel):
    # Stripped first, so a question of only spaces is rejected too.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None = None
    created_at: datetime


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    source_url: str | None = None


class RetrievalTraceItem(BaseModel):
    """Debug info for one retrieved chunk. Never contains secrets or full prompts."""

    chunk_id: str
    document_id: str
    version: int
    score: float
    used_in_answer: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation] = []
    # True when the knowledge base did not contain enough information to answer.
    insufficient_evidence: bool = False
    retrieval_trace: list[RetrievalTraceItem] = []


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    content: str
    citations: list[Citation] | None = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[MessageOut]
