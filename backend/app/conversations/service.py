"""Conversation lifecycle and history operations."""

from uuid import UUID, uuid4

from backend.app.conversations.errors import ConversationNotFoundError
from backend.app.conversations.models import ConversationRecord, ConversationTurn
from backend.app.conversations.repository import ConversationRepository


class ConversationService:
    """Create conversations and store completed exchanges."""

    def __init__(self, repository: ConversationRepository) -> None:
        self.repository = repository

    def resolve(self, conversation_id: UUID | None) -> ConversationRecord:
        if conversation_id is None:
            return self.repository.create(uuid4())
        record = self.repository.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(str(conversation_id))
        return record

    def get(self, conversation_id: UUID) -> ConversationRecord:
        record = self.repository.get(conversation_id)
        if record is None:
            raise ConversationNotFoundError(str(conversation_id))
        return record

    def append(self, conversation_id: UUID, turn: ConversationTurn) -> None:
        self.repository.append(conversation_id, turn)
