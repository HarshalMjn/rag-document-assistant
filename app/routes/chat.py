from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.database import get_db
from app.controllers.chat_controller import ChatController
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationSummary,
)
from app.schemas.common import DataResponse

router = APIRouter(tags=["chat"])


@router.get("/conversations", response_model=DataResponse[list[ConversationSummary]])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    return await ChatController.list_conversations(db)


@router.post(
    "/conversations", response_model=DataResponse[ConversationOut], status_code=201
)
async def create_conversation(db: AsyncSession = Depends(get_db)):
    return await ChatController.create_conversation(db)


@router.post(
    "/chat/{conversation_id}",
    response_model=DataResponse[ChatResponse],
    responses={
        404: {"description": "Conversation not found"},
        502: {
            "description": "The embedding or language model provider failed, "
            "or the model returned an invalid answer"
        },
    },
)
async def chat(
    conversation_id: int, request: ChatRequest, db: AsyncSession = Depends(get_db)
):
    return await ChatController.chat(db, conversation_id, request)


@router.get(
    "/chat/{conversation_id}/history",
    response_model=DataResponse[ChatHistoryResponse],
    responses={404: {"description": "Conversation not found"}},
)
async def chat_history(conversation_id: int, db: AsyncSession = Depends(get_db)):
    return await ChatController.history(db, conversation_id)
