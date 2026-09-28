"""Stored conversation models exposed by the history endpoint."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.rag.unified_models import UnifiedSource
from backend.app.routing import QueryRoute


class ConversationTurn(BaseModel):
    """One completed user/assistant exchange."""

    user_message: str
    retrieval_query: str
    assistant_answer: str
    route: QueryRoute
    sources: list[UnifiedSource]
    timestamp: datetime


class ConversationRecord(BaseModel):
    """Conversation state retained by the configured repository."""

    conversation_id: UUID
    turns: list[ConversationTurn] = Field(default_factory=list)
