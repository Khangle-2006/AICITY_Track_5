"""Concatenate JSONL manifests while preserving row order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    rows = []
    seen = set()
    for path in args.input:
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if args.dedupe and sample_id in seen:
                continue
            if sample_id:
                seen.add(sample_id)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
