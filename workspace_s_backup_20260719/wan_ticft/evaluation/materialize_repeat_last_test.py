"""Materialize repeat-last outputs for test rows, aligned to a reference output directory."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == args.view_type]
    report_rows = []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        condition_paths = numeric_pngs(Path(row["input_dir"]))
        reference_paths = numeric_pngs(args.reference_dir / output_name)
        if not condition_paths or not reference_paths:
            raise RuntimeError(f"{output_name}: missing condition/reference frames")
        anchor = read_rgb(cv2, condition_paths[-1])
        reference = read_rgb(cv2, reference_paths[0])
        if anchor is None or reference is None:
            raise RuntimeError(f"{output_name}: unreadable condition/reference frame")
        height, width = reference.shape[:2]
        if anchor.shape[:2] != (height, width):
            anchor = cv2.resize(anchor, (width, height), interpolation=cv2.INTER_AREA)
        sample_out = args.out_dir / output_name
        if sample_out.exists():
            shutil.rmtree(sample_out)
        sample_out.mkdir(parents=True)
        for frame_index in range(target_n):
            if not write_rgb(cv2, sample_out / f"{frame_index}.png", anchor):
                raise RuntimeError(f"cannot write {sample_out / f'{frame_index}.png'}")
        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report_rows.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": row["view_type"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | repeat_last", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
