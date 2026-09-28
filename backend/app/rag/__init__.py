"""Retrieval-augmented generation pipeline."""

from backend.app.rag.context import ContextBuilder
from backend.app.rag.citations import CitationBuilder
from backend.app.rag.evidence import Evidence
from backend.app.rag.factory import build_unified_rag_service
from backend.app.rag.models import RAGResponse, RetrievedChunk, SourceCitation
from backend.app.rag.service import RAGService
from backend.app.rag.unified_context import BuiltContext, UnifiedContextBuilder
from backend.app.rag.unified_models import PipelineMetadata, UnifiedRAGResponse
from backend.app.rag.unified_service import UnifiedRAGService

__all__ = [
    "BuiltContext",
    "ContextBuilder",
    "CitationBuilder",
    "Evidence",
    "build_unified_rag_service",
    "RAGResponse",
    "RAGService",
    "PipelineMetadata",
    "RetrievedChunk",
    "SourceCitation",
    "UnifiedContextBuilder",
    "UnifiedRAGResponse",
    "UnifiedRAGService",
]
