"""Construct all retrieval modes from the existing application components."""

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from time import perf_counter

from backend.app.reranking import CrossEncoderReranker
from backend.app.reranking.interfaces import Reranker
from backend.app.retrieval import (
    ChromaVectorStore,
    DenseRetriever,
    RetrievalResult,
    Retriever,
)
from backend.app.sparse import BM25Index, BM25Retriever


class MeasuredRerankingRetriever:
    """Evaluation-only wrapper exposing retrieval and reranking latency."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        candidate_count: int,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.last_timing: dict[str, float] = {}

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        retrieval_started = perf_counter()
        candidates = self.retriever.search(
            query,
            top_k=self.candidate_count,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        reranking_started = perf_counter()
        results = self.reranker.rerank(query, candidates, top_k=top_k)
        reranking_ms = (perf_counter() - reranking_started) * 1000
        self.last_timing = {
            "base_retrieval_latency_ms": round(retrieval_ms, 3),
            "reranking_latency_ms": round(reranking_ms, 3),
        }
        return list(results)


def build_evaluation_retrievers(
    settings: Settings,
    local_files_only: bool = False,
) -> dict[str, Retriever]:
    """Build four independently measured modes while sharing loaded indexes."""
    embedding = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=(
            local_files_only or settings.embedding_local_files_only
        ),
    )
    dense = DenseRetriever(
        embedding,
        ChromaVectorStore(
            settings.vector_db_path,
            settings.vector_collection_name,
        ),
    )
    bm25 = BM25Retriever(BM25Index.load(settings.bm25_index_path))
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        fusion=ReciprocalRankFusion(settings.rrf_k),
        dense_top_k=settings.hybrid_dense_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
    )
    reranked = MeasuredRerankingRetriever(
        retriever=hybrid,
        reranker=CrossEncoderReranker(
            model_name=settings.rerank_model,
            batch_size=settings.rerank_batch_size,
            local_files_only=(
                local_files_only or settings.rerank_local_files_only
            ),
        ),
        candidate_count=settings.rerank_candidates,
    )
    return {
        "Dense": dense,
        "BM25": bm25,
        "Hybrid RRF": hybrid,
        "Hybrid + Reranker": reranked,
    }
