"""Shared interface for retrieval strategies consumed by RAG."""

from typing import Protocol

from backend.app.retrieval.models import RetrievalResult


class Retriever(Protocol):
    """Structural interface implemented by dense, sparse, and hybrid retrievers."""

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return normalized ranked chunks for a query."""
