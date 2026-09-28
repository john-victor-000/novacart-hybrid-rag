"""FastAPI-facing chat request and response models."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_serializer

from backend.app.rag.unified_models import (
    PipelineMetadata,
    RetrievalDebug,
    UnifiedSource,
)
from backend.app.routing import QueryRoute


class ChatRequest(BaseModel):
    """A message with an optional existing conversation identifier."""

    query: str = Field(min_length=1)
    conversation_id: UUID | None = None
    include_debug: bool = False
    include_retrieved_chunks: bool = False

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class ChatResponse(BaseModel):
    """Stable response for frontend and evaluation clients."""

    conversation_id: UUID
    answer: str
    route: QueryRoute
    sources: list[UnifiedSource]
    metadata: PipelineMetadata
    debug: RetrievalDebug | None = None

    @model_serializer(mode="wrap")
    def omit_disabled_debug(self, handler: object) -> dict[str, object]:
        """Keep normal responses clean while preserving null source fields."""
        data = handler(self)  # type: ignore[operator]
        if self.debug is None:
            data.pop("debug", None)
        return data
