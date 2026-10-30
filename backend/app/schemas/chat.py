from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.message import MessageRole


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: int
    title: str | None = None
    created_at: datetime
    message_count: int
    last_activity: datetime


class Citation(BaseModel):
    document_id: int
    filename: str
    page: int | None = None
    chunk_id: str
    source_url: str | None = None
    text: str | None = None


class RetrievalTraceItem(BaseModel):
    chunk_id: str
    document_id: int
    version: int
    filename: str
    page: int | None = None
    chunk_index: int
    score: float
    text: str
    in_prompt: bool = False
    used_in_answer: bool = False


class ChatTimings(BaseModel):
    rewrite_ms: int
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[Citation] = []
    insufficient_evidence: bool = False
    search_query: str
    top_k: int
    history_used: int = 0
    retrieval_trace: list[RetrievalTraceItem] = []
    timings: ChatTimings | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    content: str
    citations: list[Citation] | None = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    conversation_id: int
    messages: list[MessageOut]
