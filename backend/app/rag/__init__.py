"""Baseline dense retrieval-augmented generation pipeline."""

from backend.app.rag.context import ContextBuilder
from backend.app.rag.models import RAGResponse, RetrievedChunk, SourceCitation
from backend.app.rag.service import RAGService

__all__ = [
    "ContextBuilder",
    "RAGResponse",
    "RAGService",
    "RetrievedChunk",
    "SourceCitation",
]
