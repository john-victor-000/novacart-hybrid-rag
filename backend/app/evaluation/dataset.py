"""Load and validate NovaCart evaluation questions."""

import csv
from pathlib import Path

from backend.app.evaluation.models import EvaluationQuestion

REQUIRED_COLUMNS = {"id", "question", "expected_source", "retrieval_type"}


def load_evaluation_questions(path: Path | str) -> list[EvaluationQuestion]:
    """Load document-level judgments from the project CSV."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"evaluation dataset not found: {source_path}")

    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "evaluation dataset is missing columns: "
                + ", ".join(sorted(missing))
            )
        questions = [_parse_row(row, number) for number, row in enumerate(reader, 2)]

    if not questions:
        raise ValueError("evaluation dataset contains no questions")
    identifiers = [item.question_id for item in questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation question IDs must be unique")
    return questions


def _parse_row(row: dict[str, str | None], number: int) -> EvaluationQuestion:
    values = {
        key: (row.get(key) or "").strip()
        for key in REQUIRED_COLUMNS
    }
    empty = [key for key, value in values.items() if not value]
    if empty:
        raise ValueError(
            f"evaluation row {number} has empty fields: {', '.join(sorted(empty))}"
        )
    expected = tuple(
        item.strip()
        for item in values["expected_source"].split("+")
        if item.strip()
    )
    if not expected:
        raise ValueError(f"evaluation row {number} has no expected source")
    return EvaluationQuestion(
        question_id=values["id"],
        question=values["question"],
        expected_sources=expected,
        retrieval_type=values["retrieval_type"],
    )
