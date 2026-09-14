"""Run ingestion, chunking, and embedding for NovaCart documents."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.chunking import chunk_documents
from backend.app.core.config import Settings
from backend.app.embeddings import (
    EmbeddingService,
    SentenceTransformerEmbeddingProvider,
)
from backend.app.ingestion import ingest_directory


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Ingest, chunk, and embed NovaCart documents.")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help="Number of chunks to embed per provider call.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=settings.embedding_local_files_only,
        help="Load the embedding model from the local cache only.",
    )
    args = parser.parse_args()

    records = ingest_directory(args.input_dir)
    chunks = chunk_documents(
        records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    service = EmbeddingService(provider)
    embedded = service.embed_chunks(chunks, batch_size=args.batch_size)

    sample = embedded[0] if embedded else None
    print(f"Embedded {len(embedded)} chunks")
    print(f"embedding_model={provider.model_name}")
    print(f"embedding_dimension={provider.dimension if embedded else 0}")
    if sample is not None:
        print(f"sample_chunk_id={sample.chunk_id}")
        print(
            "sample_metadata="
            f"document_id={sample.document_id}, "
            f"document_name={sample.document_name}, "
            f"document_type={sample.document_type}, "
            f"source={sample.source}, "
            f"page={sample.page if sample.page is not None else 'none'}, "
            f"section={sample.section if sample.section is not None else 'none'}"
        )
        print(f"sample_embedding_length={len(sample.embedding)}")


if __name__ == "__main__":
    main()
