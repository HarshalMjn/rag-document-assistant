from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationSummary,
    MessageOut,
)
from app.schemas.common import DataResponse
from app.services.chat_service import ChatService, ConversationNotFoundError
from app.services.gemini_service import GeminiServiceError
from app.services.llm_service import LLMServiceError


class ChatController:
    @classmethod
    async def create_conversation(
        cls, db: AsyncSession
    ) -> DataResponse[ConversationOut]:
        conversation = await ChatService.create_conversation(db)
        return DataResponse(data=ConversationOut.model_validate(conversation))

    @classmethod
    async def list_conversations(
        cls, db: AsyncSession
    ) -> DataResponse[list[ConversationSummary]]:
        return DataResponse(data=await ChatService.list_conversations(db))

    @classmethod
    async def chat(
        cls, db: AsyncSession, conversation_id: int, request: ChatRequest
    ) -> DataResponse[ChatResponse]:
        try:
            answer = await ChatService.answer(db, conversation_id, request.question)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail="Conversation not found") from e
        except GeminiServiceError as e:
            raise HTTPException(
                status_code=502, detail=f"The embedding service is unavailable: {e}"
            ) from e
        except LLMServiceError as e:
            raise HTTPException(
                status_code=502, detail=f"The language model is unavailable: {e}"
            ) from e
        return DataResponse(data=answer)

    @classmethod
    async def history(
        cls, db: AsyncSession, conversation_id: int
    ) -> DataResponse[ChatHistoryResponse]:
        try:
            messages = await ChatService.history(db, conversation_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail="Conversation not found") from e
        return DataResponse(
            data=ChatHistoryResponse(
                conversation_id=conversation_id,
                messages=[MessageOut.model_validate(m) for m in messages],
            )
        )
