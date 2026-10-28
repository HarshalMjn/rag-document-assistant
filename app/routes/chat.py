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

router = APIRouter(tags=["chat"])


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(db: AsyncSession = Depends(get_db)):
    return await ChatController.create_conversation(db)


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    return await ChatController.list_conversations(db)


@router.post(
    "/chat/{conversation_id}",
    response_model=ChatResponse,
    responses={
        404: {"description": "Conversation not found"},
        502: {"description": "The language model failed or returned an invalid answer"},
    },
)
async def chat(
    conversation_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)
):
    return await ChatController.chat(db, conversation_id, request)


@router.get(
    "/chat/{conversation_id}/history",
    response_model=ChatHistoryResponse,
    responses={404: {"description": "Conversation not found"}},
)
async def chat_history(conversation_id: str, db: AsyncSession = Depends(get_db)):
    return await ChatController.history(db, conversation_id)
