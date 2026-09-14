"""Run ingestion and chunking, then print a concise chunk summary."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from backend.app.chunking import DocumentChunk, chunk_documents
from backend.app.core.config import Settings
from backend.app.ingestion import ingest_directory


def main() -> None:
    settings = Settings()
    parser = ArgumentParser(description="Ingest and chunk NovaCart documents.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing PDF, DOCX, and CSV files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Maximum target chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Target overlap between text chunks in characters.",
    )
    args = parser.parse_args()

    records = ingest_directory(args.input_dir)
    chunks = chunk_documents(
        records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    by_file: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.document_name].append(chunk)

    print(
        f"Created {len(chunks)} chunks from {len(records)} ingested records "
        f"across {len(by_file)} files"
    )
    print(f"chunk_size={args.chunk_size} | chunk_overlap={args.chunk_overlap}")

    for document_name in sorted(by_file):
        file_chunks = by_file[document_name]
        sample_lengths = [chunk.text_length for chunk in file_chunks[:3]]
        sample_ids = [chunk.chunk_id[:12] for chunk in file_chunks[:3]]
        metadata = _format_metadata(file_chunks[0])
        print(
            f"- {document_name} | chunks={len(file_chunks)} | ids={sample_ids} | "
            f"sample_lengths={sample_lengths} | {metadata}"
        )


def _format_metadata(chunk: DocumentChunk) -> str:
    page = chunk.page if chunk.page is not None else "none"
    section = chunk.section if chunk.section is not None else "none"
    return (
        f"type={chunk.document_type} | page={page} | section={section} | "
        f"source={chunk.source}"
    )


if __name__ == "__main__":
    main()
