"""Select a balanced WTS vehicle/IP validation manifest by source subtype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_normal(row: dict) -> bool:
    return "normal" in str(row.get("source_folder", ""))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_manifest", type=Path, required=True)
    parser.add_argument("--limit_per_split", type=int, default=12)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("view_type") == "wts_vehicle_or_ip_camera"
    ]
    event_rows = [row for row in rows if not is_normal(row)]
    normal_rows = [row for row in rows if is_normal(row)]
    if len(event_rows) < args.limit_per_split or len(normal_rows) < args.limit_per_split:
        raise RuntimeError(
            "not enough WTS vehicle/IP rows: "
            f"event_ip={len(event_rows)} normal={len(normal_rows)} "
            f"required={args.limit_per_split}"
        )

    selected = []
    for event_row, normal_row in zip(
        event_rows[: args.limit_per_split],
        normal_rows[: args.limit_per_split],
    ):
        selected.extend([event_row, normal_row])

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(f"event_ip={args.limit_per_split}")
    print(f"normal={args.limit_per_split}")
    print(f"rows={len(selected)}")
    print(f"out_manifest={args.out_manifest}")


if __name__ == "__main__":
    main()
