"""Response models for the baseline RAG pipeline."""

from pydantic import BaseModel


class SourceCitation(BaseModel):
    """Source metadata copied from a retrieved chunk."""

    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    source: str
    page: int | None
    section: str | None


class RetrievedChunk(SourceCitation):
    """Optional retrieval details exposed for local debugging."""

    text: str
    score: float


class RAGResponse(BaseModel):
    """A grounded answer and its retrieved NovaCart sources."""

    answer: str
    sources: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk] | None = None
