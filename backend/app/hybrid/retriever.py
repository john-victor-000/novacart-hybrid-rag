"""Orchestrate independent dense and BM25 retrieval followed by RRF."""

from dataclasses import dataclass
import logging
from time import perf_counter

from backend.app.hybrid.fusion import ReciprocalRankFusion
from backend.app.retrieval import HybridRetrievalResult, Retriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridRetrievalResponse:
    """Independent source rankings and their fused hybrid ranking."""

    dense_results: list[RetrievalResult]
    bm25_results: list[RetrievalResult]
    hybrid_results: list[HybridRetrievalResult]


class HybridRetriever:
    """Retrieve independently with dense and BM25 strategies, then apply RRF."""

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: Retriever,
        fusion: ReciprocalRankFusion,
        dense_top_k: int,
        bm25_top_k: int,
    ) -> None:
        if dense_top_k <= 0 or bm25_top_k <= 0:
            raise ValueError("retrieval depths must be greater than zero")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion = fusion
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return only the final fused list for compatibility with RAG."""
        return list(self.retrieve(query, top_k=top_k).hybrid_results)

    def retrieve(self, query: str, top_k: int) -> HybridRetrievalResponse:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            return HybridRetrievalResponse([], [], [])

        total_started = perf_counter()

        dense_started = perf_counter()
        dense_results = self.dense_retriever.search(query, top_k=self.dense_top_k)
        dense_latency = perf_counter() - dense_started
        logger.info(
            "Hybrid dense retrieval results=%s latency=%.3fs",
            len(dense_results),
            dense_latency,
        )

        bm25_started = perf_counter()
        bm25_results = self.bm25_retriever.search(query, top_k=self.bm25_top_k)
        bm25_latency = perf_counter() - bm25_started
        logger.info(
            "Hybrid BM25 retrieval results=%s latency=%.3fs",
            len(bm25_results),
            bm25_latency,
        )

        fusion_started = perf_counter()
        hybrid_results = self.fusion.fuse(dense_results, bm25_results, top_k=top_k)
        fusion_latency = perf_counter() - fusion_started
        logger.info(
            "Hybrid RRF fusion unique_results=%s returned=%s latency=%.3fs",
            len({result.chunk_id for result in [*dense_results, *bm25_results]}),
            len(hybrid_results),
            fusion_latency,
        )
        logger.info(
            "Hybrid retrieval total_latency=%.3fs",
            perf_counter() - total_started,
        )
        return HybridRetrievalResponse(dense_results, bm25_results, hybrid_results)
