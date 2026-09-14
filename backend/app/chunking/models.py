"""Normalized chunk model built from ingested documents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieval-sized text unit with source document metadata preserved."""

    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    source: str
    page: int | None
    section: str | None
    text: str
    chunk_index: int

    @property
    def text_length(self) -> int:
        """Return the number of chunk text characters."""
        return len(self.text)
