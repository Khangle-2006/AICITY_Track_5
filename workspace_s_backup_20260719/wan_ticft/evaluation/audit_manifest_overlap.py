"""Print source-folder overlap between train and validation manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    args = parser.parse_args()

    train = read_jsonl(args.train_manifest)
    val = read_jsonl(args.val_manifest)
    for view_type in sorted({row.get("view_type", "") for row in [*train, *val]}):
        train_sources = {row.get("source_folder", "") for row in train if row.get("view_type") == view_type}
        val_sources = {row.get("source_folder", "") for row in val if row.get("view_type") == view_type}
        overlap = sorted(train_sources & val_sources)
        val_overlap = [row for row in val if row.get("view_type") == view_type and row.get("source_folder", "") in overlap]
        print(
            f"{view_type}: train_sources={len(train_sources)} "
            f"val_sources={len(val_sources)} overlap={len(overlap)} "
            f"val_overlap_rows={len(val_overlap)}"
        )
        for source, count in Counter(row.get("source_folder", "") for row in val_overlap).most_common(8):
            print(f"  {count:3d} {source}")


if __name__ == "__main__":
    main()
