from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm_client import ChatMessage
from app.config.settings import settings
from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.schemas.chat import Citation, ConversationSummary

_TITLE_LENGTH = 80


class ConversationNotFoundError(Exception):
    pass


class ConversationService:
    @classmethod
    async def create_conversation(
        cls, db: AsyncSession, title: str | None = None
    ) -> Conversation:
        conversation = Conversation(title=title)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation

    @classmethod
    async def get_conversation(
        cls, db: AsyncSession, conversation_id: int
    ) -> Conversation:
        conversation = await db.get(Conversation, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    @classmethod
    async def list_conversations(cls, db: AsyncSession) -> list[ConversationSummary]:
        """Every conversation with its message count, most recently active first."""
        last_activity = func.coalesce(
            func.max(Message.created_at), Conversation.created_at
        )
        result = await db.execute(
            select(Conversation, func.count(Message.id), last_activity)
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .group_by(Conversation.id)
            # Timestamps have 1-second precision; within a second the newest message id
            # decides, and empty conversations (NULL) sort after ones with messages.
            .order_by(
                last_activity.desc(),
                func.max(Message.id).desc(),
                Conversation.created_at.desc(),
            )
        )
        return [
            ConversationSummary(
                id=conversation.id,
                title=conversation.title,
                created_at=conversation.created_at,
                message_count=count,
                last_activity=activity,
            )
            for conversation, count, activity in result.all()
        ]

    @classmethod
    async def history(cls, db: AsyncSession, conversation_id: int) -> list[Message]:
        """Every message of one conversation, oldest first."""
        await cls.get_conversation(db, conversation_id)
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars().all())

    @classmethod
    async def recent_history(
        cls, db: AsyncSession, conversation_id: int, limit: int | None = None
    ) -> list[ChatMessage]:
        """The latest `limit` messages of one conversation (user + assistant combined), oldest first.

        Always starts with a user message.

        Defaults to HISTORY_LIMIT. Ordered by the auto-increment id, which is a strict
        insertion order even for messages saved in the same second.
        """
        limit = settings.HISTORY_LIMIT if limit is None else limit
        if limit <= 0:
            return []

        result = await db.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        history: list[ChatMessage] = [
            {"role": role.value, "content": content}
            for role, content in reversed(result.all())
        ]
        # If the window cut a pair in half, it starts with an answer whose question is gone.
        # Drop it: an answer without its question can mislead the LLM. History may then hold
        # one message fewer than `limit`.
        while history and history[0]["role"] == MessageRole.ASSISTANT:
            history.pop(0)
        return history

    @classmethod
    async def save_turn(
        cls,
        db: AsyncSession,
        conversation: Conversation,
        question: str,
        answer: str,
        citations: list[Citation],
    ) -> None:
        db.add_all(
            [
                Message(
                    conversation=conversation,
                    role=MessageRole.USER,
                    content=question,
                ),
                Message(
                    conversation=conversation,
                    role=MessageRole.ASSISTANT,
                    content=answer,
                    citations=[c.model_dump() for c in citations],
                ),
            ]
        )
        if conversation.title is None:
            conversation.title = question[:_TITLE_LENGTH]
        await db.commit()
