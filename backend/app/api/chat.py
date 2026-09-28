"""Unified automatically routed NovaCart chat endpoint."""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import Settings
from backend.app.llm import LLMProviderError
from backend.app.rag import (
    UnifiedRAGResponse,
    UnifiedRAGService,
    build_unified_rag_service,
)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """User query; retrieval strategy is selected by the backend."""

    query: str = Field(min_length=1)
    include_debug: bool = False
    include_retrieved_chunks: bool = False

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


@lru_cache
def get_rag_service() -> UnifiedRAGService:
    """Create and cache the configured unified RAG pipeline."""
    return build_unified_rag_service(Settings())


@router.post("/chat", response_model=UnifiedRAGResponse)
def chat(
    request: ChatRequest,
    service: UnifiedRAGService = Depends(get_rag_service),
) -> UnifiedRAGResponse:
    """Route, retrieve, and answer without a caller-selected retriever."""
    try:
        return service.answer(
            request.query,
            include_debug=(
                request.include_debug or request.include_retrieved_chunks
            ),
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
