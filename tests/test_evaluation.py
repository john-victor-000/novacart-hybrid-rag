import csv
import json

import pytest

from backend.app.evaluation import (
    EvaluationRunner,
    calculate_metrics,
    load_evaluation_questions,
)
from backend.app.evaluation.generation import score_generation
from backend.app.evaluation.models import EvaluationQuestion
from backend.app.retrieval import RetrievalResult
from backend.app.routing import QueryRouter


class FakeRetriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"chunk-{rank}",
                text=f"Evidence from {document}",
                score=1.0 / rank,
                document_id=f"doc-{rank}",
                document_name=document,
                document_type=document.rsplit(".", 1)[-1],
                source=f"data/raw/{document}",
                page=rank,
                section="Test",
            )
            for rank, document in enumerate(self.documents[:top_k], start=1)
        ]


def test_metrics_cover_duplicates_and_multiple_expected_sources() -> None:
    metrics = calculate_metrics(
        ["other.pdf", "faq.pdf", "faq.pdf", "return_policy.pdf"],
        ["faq.pdf", "return_policy.pdf"],
        k=5,
    )

    assert metrics.hit_rate == 1.0
    assert metrics.recall == 1.0
    assert metrics.precision == pytest.approx(0.4)
    assert metrics.reciprocal_rank == 0.5
    assert 0 < metrics.ndcg < 1


def test_runner_calculates_and_saves_measured_results(tmp_path) -> None:
    questions = [
        EvaluationQuestion(
            "E01",
            "What is the return window?",
            ("return_policy.pdf",),
            "semantic",
        )
    ]
    report = EvaluationRunner((1, 3, 5)).run(
        questions,
        {"Dense": FakeRetriever(["return_policy.pdf"])},
        {"model": "fake"},
    )
    EvaluationRunner((1, 3, 5)).save(report, tmp_path)

    summary = report["summary"][0]
    assert summary["metrics"]["1"]["hit_rate"] == 1.0
    assert summary["metrics"]["5"]["recall"] == 1.0
    assert json.loads(
        (tmp_path / "retrieval_results.json").read_text(encoding="utf-8")
    )["configuration"] == {"model": "fake"}
    assert (tmp_path / "retrieval_summary.csv").is_file()


def test_evaluation_dataset_and_golden_routes() -> None:
    questions = load_evaluation_questions(
        "data/evaluation/evaluation_questions.csv"
    )
    assert len(questions) == 10
    assert questions[3].expected_sources == (
        "products.csv",
        "warranty_guide.docx",
    )

    with open(
        "data/evaluation/golden_queries.csv",
        encoding="utf-8",
        newline="",
    ) as handle:
        golden = list(csv.DictReader(handle))
    router = QueryRouter()
    assert len(golden) == 6
    assert [
        router.route(row["query"]).route.value for row in golden
    ] == [row["expected_route"] for row in golden]


def test_generation_proxies_validate_facts_citations_and_abstention() -> None:
    scores = score_generation(
        query="What is the NCM-24 warranty?",
        answer="NCM-24 has a 24 month warranty. [Source 1]",
        evidence_texts=["Product NCM-24. Warranty months: 24."],
        cited_evidence_ids=["product:ncm-24"],
        available_evidence_ids=["product:ncm-24"],
    )
    assert scores.answer_relevance_proxy > 0
    assert scores.faithfulness_proxy == 1.0
    assert scores.citation_correct is True
    assert scores.insufficient_context_correct is None
