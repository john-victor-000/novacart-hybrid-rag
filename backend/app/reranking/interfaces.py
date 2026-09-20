"""Interface for replaceable reranking implementations."""

from typing import Protocol

from backend.app.retrieval import RerankedRetrievalResult, RetrievalResult


class Reranker(Protocol):
    """Score and reorder a bounded candidate list for one query."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RerankedRetrievalResult]:
        """Return the strongest candidates in reranker order."""
