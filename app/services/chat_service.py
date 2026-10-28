import logging

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.schemas.chat import ChatResponse, Citation, RetrievalTraceItem
from app.services.llm_service import ChatMessage, LLMService, LLMServiceError
from app.services.prompt_service import (
    INSUFFICIENT_ANSWER,
    LLMAnswer,
    PromptService,
    StandaloneQuery,
)
from app.services.qdrant_service import SearchHit
from app.services.retrieval_service import RetrievalResult, RetrievalService

logger = logging.getLogger(__name__)

_TITLE_LENGTH = 80


class ConversationNotFoundError(Exception):
    pass


class ChatService:
    @classmethod
    async def create_conversation(
        cls, db: AsyncSession, title: str | None = None
    ) -> Conversation:
        conversation = Conversation(title=title)
        db.add(conversation)
        await db.commit()
        # Load created_at, which MySQL fills in.
        await db.refresh(conversation)
        return conversation

    @classmethod
    async def answer(
        cls, db: AsyncSession, conversation_id: str, question: str
    ) -> ChatResponse:
        """Answer one question from the documents and store both turns.

        Nothing is stored if retrieval or the LLM fails, so history never holds a
        question without its answer.
        """
        conversation = await db.get(Conversation, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        # Loaded before this turn is saved, so it holds only earlier messages.
        history = await cls.recent_history(db, conversation_id)
        # Retrieval always runs fresh, on a standalone version of the question.
        search_query = await cls._standalone_query(question, history)
        retrieval = await RetrievalService.retrieve(db, search_query)
        if retrieval.evidence:
            answer, used = await cls._generate(question, retrieval.evidence, history)
        else:
            # No chunk is relevant enough, so the LLM is not called at all.
            answer, used = INSUFFICIENT_ANSWER, []

        citations = [cls._citation(hit) for hit in used]
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

        return ChatResponse(
            conversation_id=conversation.id,
            answer=answer,
            citations=citations,
            insufficient_evidence=not used,
            search_query=search_query,
            retrieval_trace=cls._trace(retrieval, used),
        )

    @classmethod
    async def recent_history(
        cls, db: AsyncSession, conversation_id: str, limit: int | None = None
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
    async def _standalone_query(cls, question: str, history: list[ChatMessage]) -> str:
        """Rewrite a follow-up ("what about sick leave?") into a standalone search query.

        Skipped when there is no history. If the rewrite fails or comes back empty, the
        original question is searched instead, so chat keeps working.
        """
        if not history:
            return question

        system, messages = PromptService.build_rewrite(question, history)
        try:
            raw = await LLMService.generate(
                system,
                messages,
                response_schema=StandaloneQuery,
                model=settings.REWRITE_MODEL,
            )
            query = StandaloneQuery.model_validate_json(raw).query.strip()
        except (LLMServiceError, ValidationError) as e:
            logger.warning(
                "Query rewrite failed, searching the original question: %s", e
            )
            return question
        return query or question

    @classmethod
    async def _generate(
        cls, question: str, evidence: list[SearchHit], history: list[ChatMessage]
    ) -> tuple[str, list[SearchHit]]:
        """Ask the LLM and return (answer, chunks it actually used).

        An answer that cites no valid source is treated as insufficient evidence:
        every answer we return must be traceable to a retrieved chunk.
        """
        prompt = PromptService.build(question, evidence, history)
        raw = await LLMService.generate(
            prompt.system_instruction, prompt.messages, response_schema=LLMAnswer
        )
        try:
            reply = LLMAnswer.model_validate_json(raw)
        except ValidationError as e:
            logger.warning("LLM returned an invalid answer: %s", e)
            raise LLMServiceError("LLM returned an invalid answer") from e

        # Only labels we handed out count; anything else is an invented citation.
        unknown = [s for s in reply.used_sources if s not in prompt.sources]
        if unknown:
            logger.warning("Dropped unknown source labels from answer: %s", unknown)
        used = [
            prompt.sources[label]
            for label in dict.fromkeys(reply.used_sources)
            if label in prompt.sources
        ]

        if not reply.answerable or not used:
            return INSUFFICIENT_ANSWER, []
        return reply.answer, used

    @staticmethod
    def _citation(hit: SearchHit) -> Citation:
        chunk = hit.payload
        return Citation(
            document_id=chunk.document_id,
            filename=chunk.filename,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            source_url=chunk.source_url,
        )

    @staticmethod
    def _trace(
        retrieval: RetrievalResult, used: list[SearchHit]
    ) -> list[RetrievalTraceItem]:
        used_ids = {hit.payload.chunk_id for hit in used}
        return [
            RetrievalTraceItem(
                chunk_id=hit.payload.chunk_id,
                document_id=hit.payload.document_id,
                version=hit.payload.version,
                score=hit.score,
                used_in_answer=hit.payload.chunk_id in used_ids,
            )
            for hit in retrieval.hits
        ]
