"""Ingest NovaCart source files and print a concise extraction summary."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from backend.app.ingestion import ingest_directory


def main() -> None:
    parser = ArgumentParser(description="Ingest NovaCart raw source documents.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing PDF, DOCX, and CSV files.",
    )
    args = parser.parse_args()

    records = ingest_directory(args.input_dir)
    by_file: dict[str, list] = defaultdict(list)
    for record in records:
        by_file[record.document_name].append(record)

    print(f"Ingested {len(records)} records from {len(by_file)} files")
    for file_name in sorted(by_file):
        file_records = by_file[file_name]
        document_type = file_records[0].document_type
        text_length = sum(record.text_length for record in file_records)
        locations = _format_locations(file_records)
        print(
            f"- {file_name} | type={document_type} | records={len(file_records)} | "
            f"text_length={text_length} | {locations}"
        )


def _format_locations(records: list) -> str:
    pages = [record.page for record in records if record.page is not None]
    rows = [
        record.section.replace("row ", "")
        for record in records
        if record.section and record.section.startswith("row ")
    ]
    sections = [
        record.section
        for record in records
        if record.section and not record.section.startswith("row ")
    ]

    if pages:
        return f"pages={min(pages)}-{max(pages)}"
    if rows:
        return f"rows={min(rows, key=int)}-{max(rows, key=int)}"
    if sections:
        return f"sections={len(records)}"
    return "locations=none"


if __name__ == "__main__":
    main()
