"""Run retrievers, calculate metrics, and persist transparent results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from backend.app.evaluation.metrics import calculate_metrics
from backend.app.evaluation.models import EvaluationQuestion
from backend.app.retrieval import Retriever, RetrievalResult


class EvaluationRunner:
    """Compare normalized retrievers against document-level judgments."""

    def __init__(self, k_values: tuple[int, ...] = (1, 3, 5)) -> None:
        normalized = tuple(sorted(set(k_values)))
        if not normalized or normalized[0] <= 0:
            raise ValueError("k values must be positive")
        self.k_values = normalized

    def run(
        self,
        questions: list[EvaluationQuestion],
        retrievers: dict[str, Retriever],
        configuration: dict[str, object] | None = None,
        warm_up: bool = False,
    ) -> dict[str, object]:
        if not questions:
            raise ValueError("questions cannot be empty")
        if not retrievers:
            raise ValueError("retrievers cannot be empty")

        details: list[dict[str, object]] = []
        summary: list[dict[str, object]] = []
        max_k = max(self.k_values)
        warmup_ms: dict[str, float] = {}
        if warm_up:
            warmup_query = questions[0].question
            for mode, retriever in retrievers.items():
                started = perf_counter()
                retriever.search(warmup_query, top_k=max_k)
                warmup_ms[mode] = round((perf_counter() - started) * 1000, 3)
        for mode, retriever in retrievers.items():
            mode_rows: list[dict[str, object]] = []
            for question in questions:
                started = perf_counter()
                results = retriever.search(question.question, top_k=max_k)
                latency_ms = (perf_counter() - started) * 1000
                row = self._query_row(
                    mode,
                    question,
                    results,
                    latency_ms,
                    getattr(retriever, "last_timing", {}),
                )
                details.append(row)
                mode_rows.append(row)
            summary.append(self._summarize(mode, mode_rows))

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "k_values": list(self.k_values),
            "configuration": configuration or {},
            "warmup_ms": warmup_ms,
            "summary": summary,
            "queries": details,
        }

    def save(self, report: dict[str, object], results_dir: Path | str) -> None:
        destination = Path(results_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "retrieval_results.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        self._write_summary_csv(
            report["summary"],  # type: ignore[arg-type]
            destination / "retrieval_summary.csv",
        )

    def _query_row(
        self,
        mode: str,
        question: EvaluationQuestion,
        results: list[RetrievalResult],
        latency_ms: float,
        timing: dict[str, float],
    ) -> dict[str, object]:
        documents = [result.document_name for result in results]
        metrics = {
            str(k): calculate_metrics(
                documents,
                question.expected_sources,
                k,
            ).as_dict()
            for k in self.k_values
        }
        return {
            "id": question.question_id,
            "query": question.question,
            "retrieval_type": question.retrieval_type,
            "expected_sources": list(question.expected_sources),
            "retrieval_mode": mode,
            "latency_ms": round(latency_ms, 3),
            "base_retrieval_latency_ms": timing.get(
                "base_retrieval_latency_ms",
                round(latency_ms, 3),
            ),
            "reranking_latency_ms": timing.get("reranking_latency_ms", 0.0),
            "expected_found": metrics[str(max(self.k_values))]["hit_rate"] == 1.0,
            "metrics": metrics,
            "retrieved": [
                _result_row(rank, result)
                for rank, result in enumerate(results, start=1)
            ],
        }

    def _summarize(
        self,
        mode: str,
        rows: list[dict[str, object]],
    ) -> dict[str, object]:
        metrics_by_k: dict[str, dict[str, float]] = {}
        for k in self.k_values:
            query_metrics = [
                row["metrics"][str(k)]  # type: ignore[index]
                for row in rows
            ]
            metrics_by_k[str(k)] = {
                metric: round(
                    mean(item[metric] for item in query_metrics),
                    6,
                )
                for metric in (
                    "hit_rate",
                    "recall",
                    "precision",
                    "reciprocal_rank",
                    "ndcg",
                )
            }
        latencies = [float(row["latency_ms"]) for row in rows]
        retrieval_latencies = [
            float(row["base_retrieval_latency_ms"]) for row in rows
        ]
        reranking_latencies = [
            float(row["reranking_latency_ms"]) for row in rows
        ]
        return {
            "retriever": mode,
            "metrics": metrics_by_k,
            "mean_latency_ms": round(mean(latencies), 3),
            "mean_retrieval_latency_ms": round(mean(retrieval_latencies), 3),
            "mean_reranking_latency_ms": round(mean(reranking_latencies), 3),
            "max_latency_ms": round(max(latencies), 3),
        }

    def _write_summary_csv(
        self,
        summary: list[dict[str, object]],
        path: Path,
    ) -> None:
        metric_names = (
            "hit_rate",
            "recall",
            "precision",
            "reciprocal_rank",
            "ndcg",
        )
        fields = ["retriever"]
        fields.extend(
            f"{metric}@{k}"
            for k in self.k_values
            for metric in metric_names
        )
        fields.extend(
            [
                "mean_retrieval_latency_ms",
                "mean_reranking_latency_ms",
                "mean_latency_ms",
                "max_latency_ms",
            ]
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in summary:
                row: dict[str, object] = {"retriever": item["retriever"]}
                for k in self.k_values:
                    values = item["metrics"][str(k)]  # type: ignore[index]
                    for metric in metric_names:
                        row[f"{metric}@{k}"] = values[metric]
                row["mean_latency_ms"] = item["mean_latency_ms"]
                row["mean_retrieval_latency_ms"] = item[
                    "mean_retrieval_latency_ms"
                ]
                row["mean_reranking_latency_ms"] = item[
                    "mean_reranking_latency_ms"
                ]
                row["max_latency_ms"] = item["max_latency_ms"]
                writer.writerow(row)


def _result_row(rank: int, result: RetrievalResult) -> dict[str, object]:
    row: dict[str, object] = {
        "rank": rank,
        "chunk_id": result.chunk_id,
        "document_name": result.document_name,
        "source": result.source,
        "page": result.page,
        "section": result.section,
        "score": result.score,
    }
    for name in (
        "dense_rank",
        "dense_score",
        "bm25_rank",
        "bm25_score",
        "rrf_rank",
        "rrf_score",
        "reranker_rank",
        "reranker_score",
    ):
        value = getattr(result, name, None)
        if value is not None:
            row[name] = value
    return row
