"""Public response models for the unified routed RAG pipeline."""

from typing import Any

from pydantic import BaseModel

from backend.app.routing import QueryRoute


class UnifiedSource(BaseModel):
    """Source metadata derived from evidence included in the prompt."""

    evidence_id: str
    source: str
    document_name: str
    document_type: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    sku: str | None = None
    retrieval_method: str | None = None
    reranker_score: float | None = None
    rrf_score: float | None = None


class EvidenceDebug(UnifiedSource):
    """Optional evidence details for local retrieval debugging."""

    text: str
    score: float | None = None
    retrieval_info: dict[str, Any]


class RetrievalDebug(BaseModel):
    """Observable routing decision and final evidence set."""

    reason: str
    confidence: float | None = None
    evidence_count: int
    evidence: list[EvidenceDebug]


class PipelineMetadata(BaseModel):
    """Timing and evidence counts produced by one unified RAG request."""

    retrieval_count: int
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float


class UnifiedRAGResponse(BaseModel):
    """One stable response shape across every retrieval route."""

    answer: str
    route: QueryRoute
    sources: list[UnifiedSource]
    metadata: PipelineMetadata
    retrieval_debug: RetrievalDebug | None = None
