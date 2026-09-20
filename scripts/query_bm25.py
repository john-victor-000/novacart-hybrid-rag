"""Run a lexical query against the persisted NovaCart BM25 index."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.retrieval import RetrievalResult
from backend.app.sparse import BM25Index, BM25Retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Search indexed NovaCart chunks with BM25.")
    parser.add_argument("query", help="Keyword, phrase, or exact identifier query.")
    parser.add_argument("--top-k", type=int, default=settings.bm25_top_k)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(settings.bm25_index_path),
    )
    args = parser.parse_args()

    if not args.index_path.is_file():
        parser.error(
            f"BM25 index not found at {args.index_path}. "
            "Run `python -m scripts.index_bm25` first."
        )

    retriever = BM25Retriever(BM25Index.load(args.index_path))
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"query={args.query}")
    print(f"top_k={args.top_k}")
    print(f"results={len(results)}")
    _print_results(results)


def _print_results(results: list[RetrievalResult]) -> None:
    for rank, result in enumerate(results, start=1):
        section = result.section if result.section is not None else "none"
        preview = " ".join(result.text.split())[:180]
        print(
            f"{rank}. bm25_score={result.score:.4f} | "
            f"chunk_id={result.chunk_id} | document={result.document_name} | "
            f"section={section}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
