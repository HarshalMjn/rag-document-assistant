import logging
import time

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm_client import ChatMessage, LLMClient, LLMClientError
from app.clients.vector_store_client import SearchHit
from app.config.settings import settings
from app.schemas.chat import ChatResponse, ChatTimings, Citation, RetrievalTraceItem
from app.services.conversation_service import ConversationService
from app.services.prompt_service import (
    INSUFFICIENT_ANSWER,
    LLMAnswer,
    PromptService,
    StandaloneQuery,
)
from app.services.retrieval_service import RetrievalResult, RetrievalService

logger = logging.getLogger(__name__)


class AnswerService:
    @classmethod
    async def answer(
        cls,
        db: AsyncSession,
        conversation_id: int,
        question: str,
        top_k: int | None = None,
    ) -> ChatResponse:
        """Answer one question from the documents and store both turns.

        Nothing is stored if retrieval or the LLM fails, so history never holds a
        question without its answer.
        """
        conversation = await ConversationService.get_conversation(db, conversation_id)
        top_k = top_k or settings.TOP_K

        started = time.perf_counter()
        # Loaded before this turn is saved, so it holds only earlier messages.
        history = await ConversationService.recent_history(db, conversation_id)
        # Retrieval always runs fresh, on a standalone version of the question.
        search_query = await cls._standalone_query(question, history)
        rewritten = time.perf_counter()
        retrieval = await RetrievalService.retrieve(db, search_query, top_k=top_k)
        retrieved = time.perf_counter()
        if retrieval.evidence:
            answer, used = await cls._generate(question, retrieval.evidence, history)
        else:
            # No chunk is relevant enough, so the LLM is not called at all.
            answer, used = INSUFFICIENT_ANSWER, []
        generated = time.perf_counter()

        citations = [cls._citation(hit) for hit in used]
        await ConversationService.save_turn(
            db, conversation, question, answer, citations
        )

        return ChatResponse(
            conversation_id=conversation.id,
            answer=answer,
            citations=citations,
            insufficient_evidence=not used,
            search_query=search_query,
            top_k=top_k,
            history_used=len(history),
            retrieval_trace=cls._trace(retrieval, used),
            timings=ChatTimings(
                rewrite_ms=_ms(started, rewritten),
                retrieval_ms=_ms(rewritten, retrieved),
                generation_ms=_ms(retrieved, generated),
                total_ms=_ms(started, generated),
            ),
        )

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
            raw = await LLMClient.generate(
                system,
                messages,
                response_schema=StandaloneQuery,
                model=settings.CHAT_REWRITE_MODEL,
            )
            query = StandaloneQuery.model_validate_json(raw).query.strip()
        except (LLMClientError, ValidationError) as e:
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
        raw = await LLMClient.generate(
            prompt.system_instruction, prompt.messages, response_schema=LLMAnswer
        )
        try:
            reply = LLMAnswer.model_validate_json(raw)
        except ValidationError as e:
            logger.warning("LLM returned an invalid answer: %s", e)
            raise LLMClientError("LLM returned an invalid answer") from e

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
            text=chunk.text,
        )

    @staticmethod
    def _trace(
        retrieval: RetrievalResult, used: list[SearchHit]
    ) -> list[RetrievalTraceItem]:
        used_ids = {hit.payload.chunk_id for hit in used}
        evidence_ids = {hit.payload.chunk_id for hit in retrieval.evidence}
        return [
            RetrievalTraceItem(
                chunk_id=hit.payload.chunk_id,
                document_id=hit.payload.document_id,
                version=hit.payload.version,
                filename=hit.payload.filename,
                page=hit.payload.page,
                chunk_index=hit.payload.chunk_index,
                score=hit.score,
                text=hit.payload.text,
                in_prompt=hit.payload.chunk_id in evidence_ids,
                used_in_answer=hit.payload.chunk_id in used_ids,
            )
            for hit in retrieval.hits
        ]


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)
