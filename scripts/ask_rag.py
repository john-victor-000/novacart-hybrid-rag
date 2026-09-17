"""Ask the baseline NovaCart RAG pipeline a question from the terminal."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.embeddings import SentenceTransformerEmbeddingProvider
from backend.app.llm import GroqLLMProvider
from backend.app.rag import ContextBuilder, RAGService
from backend.app.retrieval import ChromaVectorStore, DenseRetriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Ask the NovaCart baseline RAG pipeline.")
    parser.add_argument("query", help="Question to answer from NovaCart documents.")
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--show-chunks", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=settings.embedding_local_files_only,
    )
    parser.add_argument("--db-path", type=Path, default=Path(settings.vector_db_path))
    parser.add_argument("--collection", default=settings.vector_collection_name)
    parser.add_argument("--groq-url", default=settings.groq_base_url)
    parser.add_argument("--groq-model", default=settings.groq_model)
    args = parser.parse_args()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        local_files_only=args.local_files_only,
    )
    vector_store = ChromaVectorStore(args.db_path, args.collection)
    retriever = DenseRetriever(embedding_provider, vector_store)
    llm = GroqLLMProvider(
        settings.groq_api_key.get_secret_value(),
        args.groq_url,
        args.groq_model,
        settings.llm_timeout_seconds,
    )
    response = RAGService(retriever, ContextBuilder(), llm).answer(
        args.query,
        top_k=args.top_k,
        include_retrieved_chunks=args.show_chunks,
    )

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
