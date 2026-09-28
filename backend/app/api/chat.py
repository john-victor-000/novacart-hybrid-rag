"""Thin FastAPI routes for conversation-aware NovaCart chat."""

from functools import lru_cache
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from backend.app.chat import (
    ChatRequest,
    ChatResponse,
    ChatService,
    build_chat_service,
)
from backend.app.conversations import (
    ConversationNotFoundError,
    ConversationRecord,
)
from backend.app.core.config import Settings
from backend.app.llm import LLMProviderError
from backend.app.rag.errors import (
    RetrievalUnavailableError,
    StructuredDataUnavailableError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@lru_cache
def _cached_chat_service() -> ChatService:
    return build_chat_service(Settings())


def get_chat_service() -> ChatService:
    """Return the process-wide chat service and its conversation store."""
    try:
        return _cached_chat_service()
    except Exception as exc:
        logger.exception("Chat service initialization failed")
        raise HTTPException(
            status_code=503,
            detail="NovaCart retrieval services are unavailable",
        ) from exc


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Answer a message using automatic routing and saved conversation state."""
    try:
        return service.answer(
            request.query,
            conversation_id=request.conversation_id,
            include_debug=(
                request.include_debug or request.include_retrieved_chunks
            ),
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        ) from exc
    except LLMProviderError as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable",
        ) from exc
    except StructuredDataUnavailableError as exc:
        logger.exception("Structured retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Product data is temporarily unavailable",
        ) from exc
    except RetrievalUnavailableError as exc:
        logger.exception("Document retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Document retrieval is temporarily unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected chat failure")
        raise HTTPException(
            status_code=500,
            detail="The request could not be completed",
        ) from exc


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationRecord,
)
def get_conversation(
    conversation_id: UUID,
    service: ChatService = Depends(get_chat_service),
) -> ConversationRecord:
    """Return the stored turns for a process-local conversation."""
    try:
        return service.get_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        ) from exc
