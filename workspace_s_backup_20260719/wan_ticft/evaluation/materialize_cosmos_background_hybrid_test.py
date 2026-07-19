from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["output_name"]: row
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("video")
        }


def read_video(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def lowfreq_correct(cv2, np, base, cosmos, kernel: int, alpha: float):
    if cosmos.shape[:2] != base.shape[:2]:
        cosmos = cv2.resize(cosmos, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    base_f = base.astype(np.float32)
    cosmos_f = cosmos.astype(np.float32)
    base_low = cv2.GaussianBlur(base_f, (kernel, kernel), 0)
    cosmos_low = cv2.GaussianBlur(cosmos_f, (kernel, kernel), 0)
    return np.clip(base_f + alpha * (cosmos_low - base_low), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject Cosmos3 low-frequency background into BDD test PNG outputs.")
    parser.add_argument("--test_csv", type=Path, default=Path("/workspace/experiments/index/test_index.csv"))
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--cosmos_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--kernel", type=int, default=31)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.kernel <= 0 or args.kernel % 2 == 0:
        raise ValueError("--kernel must be a positive odd integer")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    cv2, np = import_cv2_np()
    cosmos_report = read_report(args.cosmos_report)
    with args.test_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("view_type") == "bdd"]

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for pos, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        base_paths = numeric_pngs(args.base_dir / output_name)
        if len(base_paths) != target_n:
            raise RuntimeError(f"{output_name}: base frames={len(base_paths)} expected={target_n}")
        if output_name not in cosmos_report:
            raise RuntimeError(f"{output_name}: missing Cosmos video")
        cosmos_frames, cosmos_fps = read_video(cv2, Path(cosmos_report[output_name]["video"]))
        out_sample = args.out_dir / output_name
        if out_sample.exists():
            shutil.rmtree(out_sample)
        out_sample.mkdir(parents=True, exist_ok=True)

        for index, base_path in enumerate(base_paths):
            base = read_rgb(cv2, base_path)
            if base is None:
                raise RuntimeError(f"{output_name}: cannot read {base_path}")
            cosmos_index = min(index, len(cosmos_frames) - 1)
            corrected = lowfreq_correct(cv2, np, base, cosmos_frames[cosmos_index], args.kernel, args.alpha)
            if not write_rgb(cv2, out_sample / f"{index}.png", corrected):
                raise RuntimeError(f"{output_name}: cannot write frame {index}")

        verified = verify_sample_output(out_sample, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "cosmos_frames": len(cosmos_frames),
                "cosmos_fps": f"{cosmos_fps:.6f}",
                "kernel": args.kernel,
                "alpha": f"{args.alpha:.6f}",
                "status": "ok",
            }
        )
        print(f"[{pos}/{len(rows)}] {output_name} | ok", flush=True)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "frames": sum(int(row["frames_found"]) for row in report),
                "kernel": args.kernel,
                "alpha": args.alpha,
                "status": "ok",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
