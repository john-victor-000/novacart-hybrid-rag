"""Models for chunk embeddings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk paired with its vector embedding and source metadata."""

    chunk_id: str
    text: str
    embedding: list[float]
    document_id: str
    document_name: str
    document_type: str
    source: str
    page: int | None
    section: str | None

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding vector length."""
        return len(self.embedding)
