"""Print independent dense and BM25 rankings for the same query."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.retrieval import ChromaVectorStore, DenseRetriever, RetrievalResult
from backend.app.sparse import BM25Index, BM25Retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Compare dense and BM25 rankings.")
    parser.add_argument("query", help="Query sent independently to both retrievers.")
    parser.add_argument("--top-k", type=int, default=settings.bm25_top_k)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=settings.embedding_local_files_only,
    )
    parser.add_argument("--db-path", type=Path, default=Path(settings.vector_db_path))
    parser.add_argument("--collection", default=settings.vector_collection_name)
    parser.add_argument(
        "--bm25-index-path",
        type=Path,
        default=Path(settings.bm25_index_path),
    )
    args = parser.parse_args()

    if not args.bm25_index_path.is_file():
        parser.error(
            f"BM25 index not found at {args.bm25_index_path}. "
            "Run `python -m scripts.index_bm25` first."
        )

    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    dense = DenseRetriever(
        embedding_provider,
        ChromaVectorStore(args.db_path, args.collection),
    ).search(args.query, top_k=args.top_k)
    sparse = BM25Retriever(BM25Index.load(args.bm25_index_path)).search(
        args.query,
        top_k=args.top_k,
    )

    print(f"query={args.query}")
    _print_results("DENSE RESULTS", dense)
    _print_results("BM25 RESULTS", sparse)


def _print_results(title: str, results: list[RetrievalResult]) -> None:
    print(f"\n{title}")
    for rank, result in enumerate(results, start=1):
        section = result.section if result.section is not None else "none"
        preview = " ".join(result.text.split())[:140]
        print(
            f"{rank}. score={result.score:.4f} | document={result.document_name} | "
            f"section={section} | chunk_id={result.chunk_id}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
