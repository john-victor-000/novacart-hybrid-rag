"""Load application settings from the environment and the root .env file."""

from pathlib import Path

from pydantic import Field, model_validator
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

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Keep overlap smaller than chunk size so chunking can progress."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self
