"""Embedding provider interface and local Sentence Transformers provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interface for interchangeable embedding backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the provider model name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension produced by this provider."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text inputs."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider backed by Sentence Transformers."""

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        self._model_name = model_name
        self._local_files_only = local_files_only
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._load_model()
        if self._dimension is None:
            raise RuntimeError("Embedding model dimension is unavailable")
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(float).tolist()

    def _load_model(self) -> Any:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                local_files_only=self._local_files_only,
            )
            self._dimension = int(self._model.get_sentence_embedding_dimension())
            logger.info(
                "Loaded embedding model: %s dimension=%s",
                self._model_name,
                self._dimension,
            )

        return self._model
