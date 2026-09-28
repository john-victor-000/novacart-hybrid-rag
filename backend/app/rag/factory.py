"""Construct the unified pipeline from centralized application settings."""

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from backend.app.llm import GroqLLMProvider
from backend.app.rag.unified_context import UnifiedContextBuilder
from backend.app.rag.unified_service import UnifiedRAGService
from backend.app.reranking import CrossEncoderReranker, RerankingRetriever
from backend.app.retrieval import ChromaVectorStore, DenseRetriever, Retriever
from backend.app.routing import QueryRouter
from backend.app.sparse import BM25Index, BM25Retriever
from backend.app.structured import (
    PandasProductRepository,
    ProductService,
    StructuredProductRetriever,
)


def build_unified_rag_service(settings: Settings) -> UnifiedRAGService:
    """Build reusable route dependencies without changing their implementations."""
    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=settings.embedding_local_files_only,
    )
    dense_retriever = DenseRetriever(
        embedding_provider,
        ChromaVectorStore(
            settings.vector_db_path,
            settings.vector_collection_name,
        ),
    )

    hybrid_retriever: Retriever = dense_retriever
    if settings.retrieval_mode != "dense":
        hybrid = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=BM25Retriever(
                BM25Index.load(settings.bm25_index_path)
            ),
            fusion=ReciprocalRankFusion(settings.rrf_k),
            dense_top_k=settings.hybrid_dense_top_k,
            bm25_top_k=settings.hybrid_bm25_top_k,
        )
        hybrid_retriever = RerankingRetriever(
            retriever=hybrid,
            reranker=CrossEncoderReranker(
                model_name=settings.rerank_model,
                batch_size=settings.rerank_batch_size,
                local_files_only=settings.rerank_local_files_only,
            ),
            candidate_count=settings.rerank_candidates,
            enabled=settings.rerank_enabled,
        )

    structured_retriever = StructuredProductRetriever(
        ProductService(
            PandasProductRepository.from_csv(settings.products_csv_path)
        )
    )
    llm = GroqLLMProvider(
        settings.groq_api_key.get_secret_value(),
        settings.groq_base_url,
        settings.groq_model,
        settings.llm_timeout_seconds,
    )
    hybrid_top_k = (
        settings.rerank_top_k
        if settings.rerank_enabled
        else settings.hybrid_top_k
    )
    return UnifiedRAGService(
        router=QueryRouter(),
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        structured_retriever=structured_retriever,
        context_builder=UnifiedContextBuilder(
            max_characters=settings.rag_context_max_characters,
            max_items=settings.rag_context_max_items,
        ),
        llm=llm,
        retrieval_mode=settings.retrieval_mode,
        dense_top_k=settings.retrieval_top_k,
        hybrid_top_k=hybrid_top_k,
    )
