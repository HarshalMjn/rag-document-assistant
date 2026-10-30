from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.embedding_client import EmbeddingClientError
from app.clients.llm_client import LLMClientError
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationSummary,
    MessageOut,
)
from app.schemas.common import DataResponse
from app.services.answer_service import AnswerService
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)


class ChatController:
    @classmethod
    async def create_conversation(
        cls, db: AsyncSession
    ) -> DataResponse[ConversationOut]:
        conversation = await ConversationService.create_conversation(db)
        return DataResponse(data=ConversationOut.model_validate(conversation))

    @classmethod
    async def list_conversations(
        cls, db: AsyncSession
    ) -> DataResponse[list[ConversationSummary]]:
        return DataResponse(data=await ConversationService.list_conversations(db))

    @classmethod
    async def chat(
        cls, db: AsyncSession, conversation_id: int, request: ChatRequest
    ) -> DataResponse[ChatResponse]:
        try:
            answer = await AnswerService.answer(
                db, conversation_id, request.question, top_k=request.top_k
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail="Conversation not found") from e
        except EmbeddingClientError as e:
            raise HTTPException(
                status_code=502, detail=f"The embedding service is unavailable: {e}"
            ) from e
        except LLMClientError as e:
            raise HTTPException(
                status_code=502, detail=f"The language model is unavailable: {e}"
            ) from e
        return DataResponse(data=answer)

    @classmethod
    async def history(
        cls, db: AsyncSession, conversation_id: int
    ) -> DataResponse[ChatHistoryResponse]:
        try:
            messages = await ConversationService.history(db, conversation_id)
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail="Conversation not found") from e
        return DataResponse(
            data=ChatHistoryResponse(
                conversation_id=conversation_id,
                messages=[MessageOut.model_validate(m) for m in messages],
            )
        )
