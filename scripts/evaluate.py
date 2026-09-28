"""Compare NovaCart retrieval modes using measured document-level metrics."""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.evaluation.dataset import load_evaluation_questions
from backend.app.evaluation.factory import build_evaluation_retrievers
from backend.app.evaluation.generation import score_generation
from backend.app.evaluation.models import EvaluationQuestion
from backend.app.evaluation.runner import EvaluationRunner


def main() -> None:
    settings = Settings()
    parser = ArgumentParser(description="Evaluate NovaCart retrieval quality.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/evaluation_questions.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/evaluation/results"),
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require embedding and reranker models to be cached locally.",
    )
    parser.add_argument(
        "--evaluate-generation",
        action="store_true",
        help="Also call the configured LLM for an optional proxy evaluation.",
    )
    parser.add_argument(
        "--generation-limit",
        type=int,
        default=5,
        help="Maximum questions used by optional generation evaluation.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if not Path(settings.bm25_index_path).is_file():
        parser.error(
            f"BM25 index not found at {settings.bm25_index_path}; "
            "run scripts.index_bm25 first"
        )
    if args.generation_limit <= 0:
        parser.error("--generation-limit must be greater than zero")

    questions = load_evaluation_questions(args.questions)
    retrievers = build_evaluation_retrievers(
        settings,
        local_files_only=args.local_files_only,
    )
    runner = EvaluationRunner(tuple(args.k))
    report = runner.run(
        questions,
        retrievers,
        configuration={
            "embedding_model": settings.embedding_model_name,
            "reranker_model": settings.rerank_model,
            "vector_collection": settings.vector_collection_name,
            "bm25_index": settings.bm25_index_path,
            "rrf_k": settings.rrf_k,
            "hybrid_dense_top_k": settings.hybrid_dense_top_k,
            "hybrid_bm25_top_k": settings.hybrid_bm25_top_k,
            "rerank_candidates": settings.rerank_candidates,
            "k_values": args.k,
            "warm_up_before_timing": True,
        },
        warm_up=True,
    )
    runner.save(report, args.results_dir)
    _print_summary(report)

    if args.evaluate_generation:
        _evaluate_generation(
            settings,
            questions[: args.generation_limit],
            args.results_dir,
        )

    print(f"details={args.results_dir / 'retrieval_results.json'}")
    print(f"summary={args.results_dir / 'retrieval_summary.csv'}")


def _print_summary(report: dict[str, object]) -> None:
    max_k = max(report["k_values"])  # type: ignore[arg-type]
    print(f"\nRETRIEVAL EVALUATION (K={max_k})")
    print(
        f"{'Retriever':<22} {'Hit':>7} {'Recall':>8} {'Precision':>10} "
        f"{'MRR':>7} {'NDCG':>7} {'Retrieve':>10} {'Rerank':>10} "
        f"{'Total ms':>10}"
    )
    for row in report["summary"]:  # type: ignore[union-attr]
        metrics = row["metrics"][str(max_k)]
        print(
            f"{row['retriever']:<22} "
            f"{metrics['hit_rate']:>7.3f} "
            f"{metrics['recall']:>8.3f} "
            f"{metrics['precision']:>10.3f} "
            f"{metrics['reciprocal_rank']:>7.3f} "
            f"{metrics['ndcg']:>7.3f} "
            f"{row['mean_retrieval_latency_ms']:>10.2f} "
            f"{row['mean_reranking_latency_ms']:>10.2f} "
            f"{row['mean_latency_ms']:>10.2f}"
        )


def _evaluate_generation(
    settings: Settings,
    questions: list[EvaluationQuestion],
    results_dir: Path,
) -> None:
    from backend.app.rag import build_unified_rag_service

    service = build_unified_rag_service(settings)
    rows: list[dict[str, object]] = []
    for question in questions:
        response = service.answer(question.question, include_debug=True)
        debug = response.retrieval_debug
        evidence = debug.evidence if debug is not None else []
        scores = score_generation(
            query=question.question,
            answer=response.answer,
            evidence_texts=[item.text for item in evidence],
            cited_evidence_ids=[item.evidence_id for item in response.sources],
            available_evidence_ids=[item.evidence_id for item in evidence],
        )
        rows.append(
            {
                "id": question.question_id,
                "query": question.question,
                "answer": response.answer,
                "route": response.route.value,
                "sources": [
                    source.model_dump(mode="json")
                    for source in response.sources
                ],
                "scores": scores.as_dict(),
            }
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / "generation_results.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"generation={destination}")


if __name__ == "__main__":
    main()
