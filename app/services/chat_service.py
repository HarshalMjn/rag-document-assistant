import logging

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.schemas.chat import ChatResponse, Citation, RetrievalTraceItem
from app.services.gemini_service import GeminiService, GeminiServiceError
from app.services.prompt_service import INSUFFICIENT_ANSWER, LLMAnswer, PromptService
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

        retrieval = await RetrievalService.retrieve(db, question)
        if retrieval.evidence:
            answer, used = await cls._generate(question, retrieval.evidence)
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
            retrieval_trace=cls._trace(retrieval, used),
        )

    @classmethod
    async def _generate(
        cls, question: str, evidence: list[SearchHit]
    ) -> tuple[str, list[SearchHit]]:
        """Ask the LLM and return (answer, chunks it actually used).

        An answer that cites no valid source is treated as insufficient evidence:
        every answer we return must be traceable to a retrieved chunk.
        """
        prompt = PromptService.build(question, evidence)
        raw = await GeminiService.generate(
            prompt.system_instruction, prompt.messages, response_schema=LLMAnswer
        )
        try:
            reply = LLMAnswer.model_validate_json(raw)
        except ValidationError as e:
            logger.warning("Gemini returned an invalid answer: %s", e)
            raise GeminiServiceError("Gemini returned an invalid answer") from e

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
