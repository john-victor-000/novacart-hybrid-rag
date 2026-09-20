"""Print dense, BM25, and hybrid RRF rankings for the same query."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from backend.app.retrieval import (
    ChromaVectorStore,
    DenseRetriever,
    HybridRetrievalResult,
    RetrievalResult,
)
from backend.app.sparse import BM25Index, BM25Retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Compare dense, BM25, and hybrid rankings.")
    parser.add_argument("query", help="Query sent independently to both retrievers.")
    parser.add_argument("--top-k", type=int, default=settings.hybrid_top_k)
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=settings.hybrid_dense_top_k,
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=settings.hybrid_bm25_top_k,
    )
    parser.add_argument("--rrf-k", type=int, default=settings.rrf_k)
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
    dense_retriever = DenseRetriever(
        embedding_provider,
        ChromaVectorStore(args.db_path, args.collection),
    )
    bm25_retriever = BM25Retriever(BM25Index.load(args.bm25_index_path))
    comparison = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        fusion=ReciprocalRankFusion(args.rrf_k),
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    ).retrieve(args.query, top_k=args.top_k)

    print(f"query={args.query}")
    _print_results("DENSE RESULTS", comparison.dense_results)
    _print_results("BM25 RESULTS", comparison.bm25_results)
    _print_hybrid_results(comparison.hybrid_results)


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


def _print_hybrid_results(results: list[HybridRetrievalResult]) -> None:
    print("\nHYBRID RRF RESULTS")
    for rank, result in enumerate(results, start=1):
        section = result.section if result.section is not None else "none"
        dense_rank = result.dense_rank if result.dense_rank is not None else "none"
        bm25_rank = result.bm25_rank if result.bm25_rank is not None else "none"
        preview = " ".join(result.text.split())[:140]
        print(
            f"{rank}. rrf_score={result.rrf_score:.6f} | chunk_id={result.chunk_id} | "
            f"dense_rank={dense_rank} | bm25_rank={bm25_rank} | "
            f"document={result.document_name} | section={section}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
