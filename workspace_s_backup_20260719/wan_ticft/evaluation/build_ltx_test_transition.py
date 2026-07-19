from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def transition_alpha(index: int, transition: int, curve: str) -> float:
    t = min(1.0, (index + 1) / float(transition))
    if curve == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend BDD test last-condition into early LTX frames.")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--ltx_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--transition_frames", type=int, default=16)
    parser.add_argument("--curve", choices=["linear", "smoothstep"], default="linear")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    with args.test_csv.open(encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("view_type") == "bdd"]

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        source_paths = numeric_pngs(args.ltx_dir / output_name)
        history_paths = numeric_pngs(Path(row["input_dir"]))
        if len(source_paths) != target_n:
            raise RuntimeError(f"{output_name}: LTX frames={len(source_paths)} expected={target_n}")
        if not history_paths:
            raise RuntimeError(f"{output_name}: no history frames")
        condition = read_rgb(cv2, history_paths[-1])
        if condition is None:
            raise RuntimeError(f"{output_name}: cannot read last condition frame")

        sample_out = args.out_dir / output_name
        sample_out.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(source_paths):
            destination = sample_out / f"{index}.png"
            if index >= args.transition_frames - 1:
                link_or_copy(source_path, destination)
                continue
            ltx = read_rgb(cv2, source_path)
            if ltx is None:
                raise RuntimeError(f"{output_name}: cannot read {source_path}")
            base = condition
            if base.shape[:2] != ltx.shape[:2]:
                base = cv2.resize(base, (ltx.shape[1], ltx.shape[0]), interpolation=cv2.INTER_CUBIC)
            alpha = transition_alpha(index, args.transition_frames, args.curve)
            blended = np.clip(
                (1.0 - alpha) * base.astype(np.float32) + alpha * ltx.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            if not write_rgb(cv2, destination, blended):
                raise RuntimeError(f"{output_name}: cannot write {destination}")

        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report.append(
            {
                "output_name": output_name,
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "transition_frames": args.transition_frames,
                "transition_curve": args.curve,
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "frames": sum(int(row["frames_found"]) for row in report),
                "transition_frames": args.transition_frames,
                "transition_curve": args.curve,
                "status": "ok",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
