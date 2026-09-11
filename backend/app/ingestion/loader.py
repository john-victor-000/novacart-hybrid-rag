"""Load supported documents from files or directories."""

from pathlib import Path

from backend.app.ingestion.models import IngestedDocument
from backend.app.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    parse_csv,
    parse_docx,
    parse_pdf,
)


def ingest_file(path: Path | str) -> list[IngestedDocument]:
    """Parse one supported source file into normalized documents."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".docx":
        return parse_docx(file_path)
    if suffix == ".csv":
        return parse_csv(file_path)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}")


def ingest_directory(directory: Path | str) -> list[IngestedDocument]:
    """Parse all supported source files in a directory."""
    base_path = Path(directory)
    records: list[IngestedDocument] = []

    for file_path in sorted(base_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            records.extend(ingest_file(file_path))

    return records
