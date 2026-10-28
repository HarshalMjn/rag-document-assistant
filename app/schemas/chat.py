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


class ConversationSummary(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    message_count: int
    # Time of the newest message, or created_at for an empty conversation.
    last_activity: datetime


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
    filename: str
    page: int | None = None
    chunk_index: int
    score: float
    # Scored at or above MIN_SCORE, so the chunk was sent to the LLM as evidence.
    in_prompt: bool = False
    used_in_answer: bool = False


class ChatTimings(BaseModel):
    """Milliseconds per step of one answer, for the trace view and the evaluation."""

    rewrite_ms: int
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation] = []
    # True when the knowledge base did not contain enough information to answer.
    insufficient_evidence: bool = False
    # What retrieval actually searched: the question itself, or for a follow-up its
    # standalone rewrite (e.g. "What about sick leave?" -> "sick leave days per year").
    search_query: str
    # Earlier messages sent as context: at most HISTORY_LIMIT, fewer when the window
    # dropped an answer whose question was cut off.
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
    conversation_id: str
    messages: list[MessageOut]
