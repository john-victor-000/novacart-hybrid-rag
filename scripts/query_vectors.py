"""Run dense semantic search against the persistent vector index."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.retrieval import ChromaVectorStore, DenseRetriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Search indexed NovaCart chunks.")
    parser.add_argument("query", help="Natural-language semantic search query.")
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=settings.embedding_local_files_only,
        help="Load the embedding model from the local cache only.",
    )
    parser.add_argument("--db-path", type=Path, default=Path(settings.vector_db_path))
    parser.add_argument("--collection", default=settings.vector_collection_name)
    args = parser.parse_args()

    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    store = ChromaVectorStore(args.db_path, args.collection)
    results = DenseRetriever(provider, store).search(args.query, top_k=args.top_k)

    print(f"query={args.query}")
    print(f"top_k={args.top_k}")
    print(f"results={len(results)}")
    for rank, result in enumerate(results, start=1):
        page = result.page if result.page is not None else "none"
        section = result.section if result.section is not None else "none"
        preview = " ".join(result.text.split())[:180]
        print(
            f"{rank}. score={result.score:.4f} | chunk_id={result.chunk_id} | "
            f"document={result.document_name} | type={result.document_type} | "
            f"page={page} | section={section} | source={result.source}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
