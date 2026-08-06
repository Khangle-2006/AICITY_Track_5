from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "sample_id",
    "source_folder",
    "dataset_part",
    "view_type",
    "phase",
    "history_K",
    "target_N",
    "frame_width",
    "frame_height",
    "input_dir",
    "output_dir",
    "method",
    "frames_written",
    "status",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def load_test_rows(index_dir: Path) -> list[dict[str, str]]:
    test_csv = index_dir / "test_index.csv"
    if not test_csv.exists():
        raise SystemExit(f"Missing test index: {test_csv}")
    rows = read_csv(test_csv)
    return sorted(
        rows,
        key=lambda r: (
            r.get("dataset_part", ""),
            r.get("view_type", ""),
            r.get("source_folder", ""),
            r.get("sample_id", ""),
        ),
    )
