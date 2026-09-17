"""Run ingestion, chunking, embedding, and persistent vector indexing."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.chunking import chunk_documents
from backend.app.core.config import Settings
from backend.app.embeddings import EmbeddingService, SentenceTransformerEmbeddingProvider
from backend.app.ingestion import ingest_directory
from backend.app.retrieval import ChromaVectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Index NovaCart documents in ChromaDB.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=settings.embedding_local_files_only,
        help="Load the embedding model from the local cache only.",
    )
    parser.add_argument("--db-path", type=Path, default=Path(settings.vector_db_path))
    parser.add_argument("--collection", default=settings.vector_collection_name)
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
    embedded = EmbeddingService(provider).embed_chunks(chunks, batch_size=args.batch_size)
    store = ChromaVectorStore(args.db_path, args.collection)
    newly_indexed = store.index(embedded)

    print(f"ingested_records={len(records)}")
    print(f"chunks={len(chunks)}")
    print(f"embedded_chunks={len(embedded)}")
    print(f"newly_indexed={newly_indexed}")
    print(f"collection={args.collection}")
    print(f"db_path={args.db_path.resolve()}")
    print(f"total_indexed={store.count()}")


if __name__ == "__main__":
    main()
