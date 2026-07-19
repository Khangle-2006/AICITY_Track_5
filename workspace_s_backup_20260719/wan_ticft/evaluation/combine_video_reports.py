"""Concatenate video report CSVs while preserving a stable union of columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    fields: list[str] = []
    rows: list[dict[str, str]] = []
    seen = set()
    for path in args.input:
        path_fields, path_rows = read_rows(path)
        for field in path_fields:
            if field not in fields:
                fields.append(field)
        for row in path_rows:
            sample_id = row.get("sample_id")
            if args.dedupe and sample_id in seen:
                continue
            if sample_id:
                seen.add(sample_id)
            rows.append(row)

    if not fields:
        raise RuntimeError("no CSV fields found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"rows={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
