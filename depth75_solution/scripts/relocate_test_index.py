from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite test_index.csv input_dir values for a local Track 5 data root."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def locate_input(data_root: Path, source_folder: str) -> Path:
    direct = data_root / source_folder / "input"
    if direct.is_dir():
        return direct.resolve()

    matches = [path for path in data_root.rglob(source_folder) if path.is_dir() and (path / "input").is_dir()]
    if not matches:
        raise FileNotFoundError(f"cannot find {source_folder}/input below {data_root}")
    if len(matches) > 1:
        rendered = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"ambiguous source folder {source_folder}:\n  {rendered}")
    return (matches[0] / "input").resolve()


def main() -> None:
    args = parse_args()
    with args.template.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or "source_folder" not in fields or "input_dir" not in fields:
        raise RuntimeError(f"invalid Track 5 index template: {args.template}")

    cache: dict[str, Path] = {}
    for row in rows:
        source_folder = row["source_folder"]
        cache.setdefault(source_folder, locate_input(args.data_root, source_folder))
        row["input_dir"] = str(cache[source_folder])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"index={args.output}")
    print(f"samples={len(rows)}")


if __name__ == "__main__":
    main()
