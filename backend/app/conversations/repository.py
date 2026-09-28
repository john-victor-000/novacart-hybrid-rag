"""Repository abstraction for lightweight conversation persistence."""

from __future__ import annotations

from threading import RLock
from typing import Protocol
from uuid import UUID

from backend.app.conversations.models import ConversationRecord, ConversationTurn


class ConversationRepository(Protocol):
    """Storage contract that can later be backed by Redis or a database."""

    def create(self, conversation_id: UUID) -> ConversationRecord:
        """Create and return an empty conversation."""

    def get(self, conversation_id: UUID) -> ConversationRecord | None:
        """Return a conversation when it exists."""

    def append(self, conversation_id: UUID, turn: ConversationTurn) -> None:
        """Append a completed turn atomically."""


class InMemoryConversationRepository:
    """Thread-safe conversation storage for one API process."""

    def __init__(self) -> None:
        self._records: dict[UUID, ConversationRecord] = {}
        self._lock = RLock()

    def create(self, conversation_id: UUID) -> ConversationRecord:
        with self._lock:
            record = ConversationRecord(conversation_id=conversation_id)
            self._records[conversation_id] = record
            return record.model_copy(deep=True)

    def get(self, conversation_id: UUID) -> ConversationRecord | None:
        with self._lock:
            record = self._records.get(conversation_id)
            return record.model_copy(deep=True) if record is not None else None

    def append(self, conversation_id: UUID, turn: ConversationTurn) -> None:
        with self._lock:
            self._records[conversation_id].turns.append(turn)
