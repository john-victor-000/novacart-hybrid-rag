"""Document ingestion utilities."""

from backend.app.ingestion.loader import ingest_directory, ingest_file
from backend.app.ingestion.models import IngestedDocument

__all__ = ["IngestedDocument", "ingest_directory", "ingest_file"]
