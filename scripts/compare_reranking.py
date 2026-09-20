"""Compare hybrid RRF candidates with cross-encoder reranked results."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from backend.app.reranking import CrossEncoderReranker
from backend.app.retrieval import (
    ChromaVectorStore,
    DenseRetriever,
    HybridRetrievalResult,
    RerankedRetrievalResult,
)
from backend.app.sparse import BM25Index, BM25Retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Compare hybrid and reranked results.")
    parser.add_argument("query", help="Query sent through hybrid retrieval.")
    parser.add_argument(
        "--candidates",
        type=int,
        default=settings.rerank_candidates,
    )
    parser.add_argument("--top-k", type=int, default=settings.rerank_top_k)
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
    parser.add_argument("--rerank-model", default=settings.rerank_model)
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=settings.rerank_batch_size,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Require both embedding and reranker models to exist locally.",
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
    if args.candidates <= 0 or args.top_k <= 0:
        parser.error("--candidates and --top-k must be greater than zero")

    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=(
            args.local_files_only or settings.embedding_local_files_only
        ),
    )
    hybrid_retriever = HybridRetriever(
        dense_retriever=DenseRetriever(
            embedding_provider,
            ChromaVectorStore(args.db_path, args.collection),
        ),
        bm25_retriever=BM25Retriever(BM25Index.load(args.bm25_index_path)),
        fusion=ReciprocalRankFusion(args.rrf_k),
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    )
    hybrid_results = hybrid_retriever.search(args.query, top_k=args.candidates)
    reranked_results = CrossEncoderReranker(
        model_name=args.rerank_model,
        batch_size=args.rerank_batch_size,
        local_files_only=(
            args.local_files_only or settings.rerank_local_files_only
        ),
    ).rerank(args.query, hybrid_results, top_k=args.top_k)

    print(f"query={args.query}")
    _print_hybrid(hybrid_results)
    _print_reranked(reranked_results)


def _print_hybrid(results: list[HybridRetrievalResult]) -> None:
    print("\nHYBRID RRF RANKING")
    for rrf_rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())[:140]
        print(
            f"{rrf_rank}. chunk_id={result.chunk_id} | "
            f"document={result.document_name} | rrf_rank={rrf_rank} | "
            f"rrf_score={result.rrf_score:.6f} | reranker_rank=none | "
            "reranker_score=none"
        )
        print(f"   {preview}")


def _print_reranked(results: list[RerankedRetrievalResult]) -> None:
    print("\nRERANKED RANKING")
    for result in results:
        rrf_score = (
            f"{result.rrf_score:.6f}"
            if result.rrf_score is not None
            else "none"
        )
        preview = " ".join(result.text.split())[:140]
        print(
            f"{result.reranker_rank}. chunk_id={result.chunk_id} | "
            f"document={result.document_name} | rrf_rank={result.rrf_rank} | "
            f"rrf_score={rrf_score} | reranker_rank={result.reranker_rank} | "
            f"reranker_score={result.reranker_score:.6f}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
