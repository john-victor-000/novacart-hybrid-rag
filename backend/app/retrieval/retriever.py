"""Dense retrieval orchestration independent of the vector database."""

from __future__ import annotations

import logging
from time import perf_counter

from backend.app.embeddings.providers import EmbeddingProvider
from backend.app.retrieval.models import RetrievalResult
from backend.app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Embed a query and search a replaceable vector store."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.provider = provider
        self.vector_store = vector_store

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return the most similar indexed chunks for a user query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            return []

        started = perf_counter()
        query_embedding = self.provider.embed_texts([query])[0]
        if len(query_embedding) != self.provider.dimension:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"expected {self.provider.dimension}, got {len(query_embedding)}"
            )

        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.info(
            "Retrieved %s results for dense query in %.2fs",
            len(results),
            perf_counter() - started,
        )
        return results
