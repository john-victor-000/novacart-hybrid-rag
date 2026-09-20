"""Run ingestion and chunking, then persist a BM25 sparse index."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.chunking import chunk_documents
from backend.app.core.config import Settings
from backend.app.ingestion import ingest_directory
from backend.app.sparse import BM25Index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Build the NovaCart BM25 index.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(settings.bm25_index_path),
    )
    parser.add_argument("--k1", type=float, default=settings.bm25_k1)
    parser.add_argument("--b", type=float, default=settings.bm25_b)
    args = parser.parse_args()

    records = ingest_directory(args.input_dir)
    chunks = chunk_documents(
        records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    index = BM25Index.build(chunks, k1=args.k1, b=args.b)
    index.save(args.index_path)

    print(f"ingested_records={len(records)}")
    print(f"indexed_chunks={index.count}")
    print(f"index_path={args.index_path.resolve()}")
    print(f"k1={index.k1}")
    print(f"b={index.b}")


if __name__ == "__main__":
    main()
