"""Embedding service that preserves chunk metadata."""

from __future__ import annotations

import logging
from time import perf_counter

from backend.app.chunking.models import DocumentChunk
from backend.app.embeddings.models import EmbeddedChunk
from backend.app.embeddings.providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embed chunks with an interchangeable provider."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Embed a single chunk."""
        return self.embed_chunks([chunk], batch_size=1)[0]

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
    ) -> list[EmbeddedChunk]:
        """Embed chunks in batches while preserving chunk metadata."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        started = perf_counter()
        dimension = self.provider.dimension
        embedded: list[EmbeddedChunk] = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            non_empty = [chunk for chunk in batch if chunk.text.strip()]
            vectors_by_chunk_id: dict[str, list[float]] = {}

            if non_empty:
                vectors = self.provider.embed_texts([chunk.text for chunk in non_empty])
                self._validate_dimensions(vectors, dimension)
                vectors_by_chunk_id = {
                    chunk.chunk_id: vector
                    for chunk, vector in zip(non_empty, vectors, strict=True)
                }

            for chunk in batch:
                vector = vectors_by_chunk_id.get(chunk.chunk_id)
                if vector is None:
                    vector = [0.0] * dimension
                embedded.append(self._build_record(chunk, vector))

        elapsed = perf_counter() - started
        logger.info(
            "Embedded %s chunks with model=%s dimension=%s in %.2fs",
            len(chunks),
            self.provider.model_name,
            dimension,
            elapsed,
        )
        return embedded

    def _validate_dimensions(
        self,
        vectors: list[list[float]],
        expected_dimension: int,
    ) -> None:
        for vector in vectors:
            if len(vector) != expected_dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {expected_dimension}, got {len(vector)}"
                )

    def _build_record(
        self,
        chunk: DocumentChunk,
        embedding: list[float],
    ) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            embedding=embedding,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            document_type=chunk.document_type,
            source=chunk.source,
            page=chunk.page,
            section=chunk.section,
        )
