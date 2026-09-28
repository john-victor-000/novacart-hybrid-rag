"""Conversation-aware wrapper around the unified RAG service."""

from datetime import datetime, timezone
import logging
from time import perf_counter
from uuid import UUID

from backend.app.chat.models import ChatResponse
from backend.app.conversations import (
    ConversationRecord,
    ConversationService,
    ConversationTurn,
    FollowUpRewriter,
)
from backend.app.rag import UnifiedRAGService

logger = logging.getLogger(__name__)


class ChatService:
    """Resolve history, rewrite a narrow follow-up, and run unified RAG."""

    def __init__(
        self,
        rag_service: UnifiedRAGService,
        conversations: ConversationService,
        followups: FollowUpRewriter,
    ) -> None:
        self.rag_service = rag_service
        self.conversations = conversations
        self.followups = followups

    def answer(
        self,
        query: str,
        conversation_id: UUID | None = None,
        include_debug: bool = False,
    ) -> ChatResponse:
        started = perf_counter()
        conversation = self.conversations.resolve(conversation_id)
        rewrite = self.followups.rewrite(query, conversation.turns)
        request_id = str(conversation.conversation_id)
        logger.info(
            "Chat request conversation_id=%s rewritten=%s",
            request_id,
            rewrite.rewritten,
        )

        result = self.rag_service.answer(
            rewrite.query,
            include_debug=include_debug,
            request_id=request_id,
        )
        turn = ConversationTurn(
            user_message=query,
            retrieval_query=rewrite.query,
            assistant_answer=result.answer,
            route=result.route,
            sources=result.sources,
            timestamp=datetime.now(timezone.utc),
        )
        self.conversations.append(conversation.conversation_id, turn)
        logger.info(
            "Chat completed conversation_id=%s route=%s retrieval_items=%s "
            "total_latency=%.3fs",
            request_id,
            result.route.value,
            result.metadata.retrieval_count,
            perf_counter() - started,
        )
        return ChatResponse(
            conversation_id=conversation.conversation_id,
            answer=result.answer,
            route=result.route,
            sources=result.sources,
            metadata=result.metadata,
            debug=result.retrieval_debug if include_debug else None,
        )

    def get_conversation(self, conversation_id: UUID) -> ConversationRecord:
        return self.conversations.get(conversation_id)
