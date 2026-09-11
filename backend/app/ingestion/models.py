"""Normalized document model used by the ingestion phase."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestedDocument:
    """Text extracted from one source location with normalized metadata."""

    document_id: str
    document_name: str
    document_type: str
    source: str
    page: int | None
    section: str | None
    text: str

    @property
    def text_length(self) -> int:
        """Return the number of extracted text characters."""
        return len(self.text)
