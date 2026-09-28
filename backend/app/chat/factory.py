"""Build the conversation-aware chat application service."""

from backend.app.chat.service import ChatService
from backend.app.conversations import (
    ConversationService,
    FollowUpRewriter,
    InMemoryConversationRepository,
)
from backend.app.core.config import Settings
from backend.app.rag import build_unified_rag_service


def build_chat_service(settings: Settings) -> ChatService:
    """Construct chat orchestration while reusing the Stage A pipeline."""
    return ChatService(
        rag_service=build_unified_rag_service(settings),
        conversations=ConversationService(InMemoryConversationRepository()),
        followups=FollowUpRewriter(),
    )
