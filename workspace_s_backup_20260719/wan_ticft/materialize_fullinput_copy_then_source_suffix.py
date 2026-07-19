"""Copy known input frames, then take the suffix from an existing prediction source."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_lookup(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        out[row["sample_id"]] = row
        out[row["source_folder"]] = row
    return out


def resize_720p(frame):
    if frame.shape[:2] == (720, 1280):
        return frame
    return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lookup = manifest_lookup(args.manifest)
    rows = read_csv(args.source_report)
    report: list[dict[str, str | int]] = []
    for pos, row in enumerate(rows, start=1):
        output_name = row["output_name"]
        target_n = int(row["target_N"])
        sample_out = args.out_dir / output_name
        existing = verify_sample_output(sample_out, target_n)
        if existing.ok and not args.clean:
            report.append(
                {
                    "output_name": output_name,
                    "sample_id": row["sample_id"],
                    "source_folder": row["source_folder"],
                    "view_type": row["view_type"],
                    "target_N": target_n,
                    "source_choice": "copy_input_then_" + row.get("source_choice", "source"),
                    "source_dir": str(sample_out),
                    "input_frames": "skip",
                    "suffix_frames": "skip",
                    "frames_found": existing.frames_found,
                    "status": "ok",
                    "notes": "skip_existing",
                }
            )
            print(f"[{pos}/{len(rows)}] {output_name} | skip(existing)", flush=True)
            continue

        manifest_row = lookup.get(row["sample_id"]) or lookup.get(row["source_folder"])
        if manifest_row is None:
            raise RuntimeError(f"{output_name}: missing manifest row")
        input_paths = numeric_pngs(Path(manifest_row["input_dir"]))
        source_paths = numeric_pngs(Path(row["source_dir"]))
        if not input_paths:
            raise RuntimeError(f"{output_name}: no input frames")
        if not source_paths:
            raise RuntimeError(f"{output_name}: no source frames in {row['source_dir']}")
        sample_out.mkdir(parents=True, exist_ok=True)
        copied = min(len(input_paths), target_n)
        suffix = max(0, target_n - copied)
        last_source = source_paths[-1]
        for index in range(copied):
            frame = resize_720p(read_rgb(cv2, input_paths[index]))
            write_rgb(cv2, sample_out / f"{index}.png", frame)
        for index in range(copied, target_n):
            src = source_paths[index] if index < len(source_paths) else last_source
            frame = resize_720p(read_rgb(cv2, src))
            write_rgb(cv2, sample_out / f"{index}.png", frame)
        verify = verify_sample_output(sample_out, target_n)
        if not verify.ok:
            raise RuntimeError(f"{output_name}: verify failed {verify.notes}")
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "source_folder": row["source_folder"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "source_choice": "copy_input_then_" + row.get("source_choice", "source"),
                "source_dir": str(sample_out),
                "input_frames": len(input_paths),
                "suffix_frames": suffix,
                "frames_found": verify.frames_found,
                "status": "ok",
                "notes": "",
            }
        )
        print(
            f"[{pos}/{len(rows)}] {output_name} input={len(input_paths)} target={target_n} suffix={suffix} | ok",
            flush=True,
        )

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "output_name",
            "sample_id",
            "source_folder",
            "view_type",
            "target_N",
            "source_choice",
            "source_dir",
            "input_frames",
            "suffix_frames",
            "frames_found",
            "status",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
