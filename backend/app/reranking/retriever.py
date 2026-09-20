"""Optional reranking wrapper for any normalized candidate retriever."""

import logging

from backend.app.reranking.interfaces import Reranker
from backend.app.retrieval import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


class RerankingRetriever:
    """Retrieve a wider candidate set and optionally rerank it."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        candidate_count: int,
        enabled: bool = True,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("rerank candidate count must be greater than zero")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.enabled = enabled

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            return []
        if not self.enabled:
            logger.info("Reranking disabled; returning retrieval order")
            return self.retriever.search(query, top_k=top_k)

        candidates = self.retriever.search(query, top_k=self.candidate_count)
        return list(self.reranker.rerank(query, candidates, top_k=top_k))
