"""Route test output folders by phase between existing candidate directories."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import numeric_pngs
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--default_dir", type=Path, required=True)
    parser.add_argument("--default_label", default="default")
    parser.add_argument("--phase_dir", action="append", default=[], help="phase=/path/to/dir=label")
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    phase_map: dict[str, tuple[Path, str]] = {}
    for item in args.phase_dir:
        phase, directory, label = item.split("=", 2)
        phase_map[str(phase)] = (Path(directory), label)

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        phase = str(row.get("phase", ""))
        source_root, label = phase_map.get(phase, (args.default_dir, args.default_label))
        source_dir = source_root / output_name
        verified = verify_sample_output(source_dir, target_n)
        if not verified.ok:
            if source_root != args.default_dir:
                source_root, label = args.default_dir, args.default_label
                source_dir = source_root / output_name
                verified = verify_sample_output(source_dir, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: source verification failed: {verified.notes}")
        destination = args.out_dir / output_name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for png in numeric_pngs(source_dir):
            shutil.copy2(png, destination / f"{int(png.stem)}.png")
        verified = verify_sample_output(destination, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: output verification failed: {verified.notes}")
        report_rows.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "phase": phase,
                "target_N": target_n,
                "source_choice": label,
                "source_dir": str(source_dir),
                "frames_found": verified.frames_found,
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | {label}", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
