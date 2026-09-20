"""Load application settings from the environment and the root .env file."""

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment variables override .env values and built-in defaults."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "NovaCart Hybrid RAG Knowledge Assistant"
    app_debug: bool = False
    chunk_size: int = Field(default=900, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=16, gt=0)
    embedding_local_files_only: bool = False
    vector_db_path: str = "data/vector_store"
    vector_collection_name: str = "novacart_chunks"
    retrieval_top_k: int = Field(default=5, gt=0)
    bm25_index_path: str = "data/bm25_index.json"
    bm25_top_k: int = Field(default=5, gt=0)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    rrf_k: int = Field(default=60, gt=0)
    hybrid_dense_top_k: int = Field(default=10, gt=0)
    hybrid_bm25_top_k: int = Field(default=10, gt=0)
    hybrid_top_k: int = Field(default=5, gt=0)
    rag_retrieval_mode: str = "dense"
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = Field(default=20, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    rerank_batch_size: int = Field(default=8, gt=0)
    rerank_local_files_only: bool = False
    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Keep overlap smaller than chunk size so chunking can progress."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.rag_retrieval_mode not in {"dense", "hybrid"}:
            raise ValueError("RAG_RETRIEVAL_MODE must be 'dense' or 'hybrid'")
        if self.rerank_candidates < self.rerank_top_k:
            raise ValueError(
                "RERANK_CANDIDATES must be greater than or equal to RERANK_TOP_K"
            )
        return self
