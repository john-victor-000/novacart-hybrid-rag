"""RAG chat endpoint with configurable dense or hybrid retrieval."""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from backend.app.llm import GroqLLMProvider, LLMProviderError
from backend.app.rag import ContextBuilder, RAGResponse, RAGService
from backend.app.retrieval import ChromaVectorStore, DenseRetriever, Retriever
from backend.app.sparse import BM25Index, BM25Retriever

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """User input for the baseline RAG endpoint."""

    query: str = Field(min_length=1)
    include_retrieved_chunks: bool = False

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


@lru_cache
def get_rag_service() -> RAGService:
    """Create and cache the configured local retrieval and RAG dependencies."""
    settings = Settings()
    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=settings.embedding_local_files_only,
    )
    vector_store = ChromaVectorStore(
        settings.vector_db_path,
        settings.vector_collection_name,
    )
    dense_retriever = DenseRetriever(embedding_provider, vector_store)
    retriever: Retriever = dense_retriever
    if settings.rag_retrieval_mode == "hybrid":
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=BM25Retriever(BM25Index.load(settings.bm25_index_path)),
            fusion=ReciprocalRankFusion(settings.rrf_k),
            dense_top_k=settings.hybrid_dense_top_k,
            bm25_top_k=settings.hybrid_bm25_top_k,
        )
    llm = GroqLLMProvider(
        settings.groq_api_key.get_secret_value(),
        settings.groq_base_url,
        settings.groq_model,
        settings.llm_timeout_seconds,
    )
    return RAGService(retriever, ContextBuilder(), llm)


@router.post("/chat", response_model=RAGResponse)
def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """Answer a question using configured retrieval and grounded generation."""
    settings = Settings()
    top_k = (
        settings.hybrid_top_k
        if settings.rag_retrieval_mode == "hybrid"
        else settings.retrieval_top_k
    )
    try:
        return service.answer(
            request.query,
            top_k=top_k,
            include_retrieved_chunks=request.include_retrieved_chunks,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
