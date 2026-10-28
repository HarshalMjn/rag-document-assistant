import logging

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationOut,
    MessageOut,
)
from app.services.chat_service import ChatService, ConversationNotFoundError
from app.services.gemini_service import GeminiServiceError
from app.services.llm_service import LLMServiceError

logger = logging.getLogger(__name__)


class ChatController:
    @classmethod
    async def create_conversation(cls, db: AsyncSession) -> ConversationOut:
        conversation = await ChatService.create_conversation(db)
        return ConversationOut.model_validate(conversation)

    @classmethod
    async def chat(
        cls, db: AsyncSession, conversation_id: str, request: ChatRequest
    ) -> ChatResponse:
        try:
            return await ChatService.answer(db, conversation_id, request.question)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail="Conversation not found") from e
        except (GeminiServiceError, LLMServiceError) as e:
            # The message is already safe (status only, never the key or provider text).
            raise HTTPException(
                status_code=502, detail=f"The language model is unavailable: {e}"
            ) from e

    @classmethod
    async def history(
        cls, db: AsyncSession, conversation_id: str
    ) -> ChatHistoryResponse:
        try:
            messages = await ChatService.history(db, conversation_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail="Conversation not found") from e
        return ChatHistoryResponse(
            conversation_id=conversation_id,
            messages=[MessageOut.model_validate(m) for m in messages],
        )
