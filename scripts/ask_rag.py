"""Ask NovaCart RAG with configurable dense or hybrid retrieval."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from backend.app.llm import GroqLLMProvider
from backend.app.rag import ContextBuilder, RAGService
from backend.app.retrieval import ChromaVectorStore, DenseRetriever, Retriever
from backend.app.sparse import BM25Index, BM25Retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Ask NovaCart with dense or hybrid RAG.")
    parser.add_argument("query", help="Question to answer from NovaCart documents.")
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "hybrid"),
        default=settings.rag_retrieval_mode,
    )
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
    parser.add_argument("--show-chunks", action="store_true")
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
    parser.add_argument("--groq-url", default=settings.groq_base_url)
    parser.add_argument("--groq-model", default=settings.groq_model)
    args = parser.parse_args()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    vector_store = ChromaVectorStore(args.db_path, args.collection)
    dense_retriever = DenseRetriever(embedding_provider, vector_store)
    retriever: Retriever = dense_retriever
    if args.retrieval_mode == "hybrid":
        if not args.bm25_index_path.is_file():
            parser.error(
                f"BM25 index not found at {args.bm25_index_path}. "
                "Run `python -m scripts.index_bm25` first."
            )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=BM25Retriever(BM25Index.load(args.bm25_index_path)),
            fusion=ReciprocalRankFusion(args.rrf_k),
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
        )
    llm = GroqLLMProvider(
        settings.groq_api_key.get_secret_value(),
        args.groq_url,
        args.groq_model,
        settings.llm_timeout_seconds,
    )
    top_k = (
        args.top_k
        if args.top_k is not None
        else (
            settings.hybrid_top_k
            if args.retrieval_mode == "hybrid"
            else settings.retrieval_top_k
        )
    )
    response = RAGService(retriever, ContextBuilder(), llm).answer(
        args.query,
        top_k=top_k,
        include_retrieved_chunks=args.show_chunks,
    )

    print(f"retrieval_mode={args.retrieval_mode}")
    print(f"answer={response.answer}")
    print(f"sources={len(response.sources)}")
    for source in response.sources:
        page = source.page if source.page is not None else "none"
        section = source.section if source.section is not None else "none"
        print(
            f"- {source.document_name} | page={page} | section={section} | "
            f"chunk_id={source.chunk_id} | source={source.source}"
        )

    if response.retrieved_chunks is not None:
        print(f"retrieved_chunks={len(response.retrieved_chunks)}")
        for chunk in response.retrieved_chunks:
            preview = " ".join(chunk.text.split())[:180]
            print(f"- score={chunk.score:.4f} | chunk_id={chunk.chunk_id} | {preview}")


if __name__ == "__main__":
    main()
