"""Conversation-domain errors."""


class ConversationNotFoundError(LookupError):
    """A supplied conversation identifier is unknown to this process."""
