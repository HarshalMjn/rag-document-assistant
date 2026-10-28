import logging

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import ChatRequest, ChatResponse, ConversationOut
from app.services.chat_service import ChatService, ConversationNotFoundError
from app.services.gemini_service import GeminiServiceError

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
        except GeminiServiceError as e:
            # The message is already safe (status only, never the key or provider text).
            raise HTTPException(
                status_code=502, detail=f"The language model is unavailable: {e}"
            ) from e
