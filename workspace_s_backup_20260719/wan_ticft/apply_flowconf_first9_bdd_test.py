"""Apply 9-frame flow-confidence composite to a full BDD PNG tree.

The learned flow-confidence branch only predicts the first nine future frames.
This materializer uses it only where available and copies the base candidate for
the rest of the horizon, so submission frame counts stay valid.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_flow_report(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for row in read_csv(path):
        if row.get("status") == "ok" and row.get("video"):
            out[row["sample_id"]] = Path(row["video"])
    return out


def read_video_frames(video: Path, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_alpha(alpha_dir: Path, index: int, shape: tuple[int, int], threshold: float, softness: float, blur: float) -> np.ndarray:
    gray = cv2.imread(str(alpha_dir / f"alpha_{index:03d}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(shape, dtype=np.float32)
    if gray.shape[:2] != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = gray.astype(np.float32) / 255.0
    mask = np.clip((alpha - threshold) / max(softness, 1e-6), 0.0, 1.0)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--base_root", type=Path, required=True)
    parser.add_argument("--flow_report", type=Path, required=True)
    parser.add_argument("--alpha_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--softness", type=float, default=0.08)
    parser.add_argument("--blur", type=float, default=2.0)
    parser.add_argument("--max_flow_frames", type=int, default=9)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2.setNumThreads(4)
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    flow_report = read_flow_report(args.flow_report)
    rows = [row for row in read_csv(args.base_report) if row.get("status") == "ok" and row.get("view_type") == "bdd"]
    report: list[dict[str, str | int]] = []
    diag: list[dict[str, str | int | float]] = []
    for pos, row in enumerate(rows, start=1):
        output_name = row["output_name"]
        sample_id = row["sample_id"]
        target_n = int(row["target_N"])
        base_dir = args.base_root / output_name
        out_dir = args.out_root / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        flow_frames = read_video_frames(flow_report[sample_id], args.max_flow_frames) if sample_id in flow_report else []
        blend_pixels = []
        written = 0
        for index in range(target_n):
            bgr = cv2.imread(str(base_dir / f"{index}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = base
            if index < len(flow_frames):
                flow = flow_frames[index]
                if flow.shape[:2] != base.shape[:2]:
                    flow = cv2.resize(flow, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
                mask = load_alpha(args.alpha_root / sample_id, index, base.shape[:2], args.threshold, args.softness, args.blur)
                weight = np.clip(mask * args.strength, 0.0, 1.0)[:, :, None]
                out = np.clip(base.astype(np.float32) * (1.0 - weight) + flow.astype(np.float32) * weight, 0, 255).astype(np.uint8)
                blend_pixels.append(float(mask.mean()))
            ok = cv2.imwrite(str(out_dir / f"{index}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            written += int(bool(ok))
        status = "ok" if written == target_n else "bad"
        report.append(
            {
                "output_name": output_name,
                "sample_id": sample_id,
                "source_folder": output_name,
                "dataset_part": row.get("dataset_part", ""),
                "view_type": "bdd",
                "target_N": target_n,
                "source_choice": f"flowconf_first9_t{args.threshold:.2f}_s{args.strength:.2f}",
                "source_dir": str(out_dir),
                "frames_found": written,
                "status": status,
                "notes": "",
            }
        )
        diag.append(
            {
                "output_name": output_name,
                "sample_id": sample_id,
                "flow_frames": len(flow_frames),
                "blend_alpha_mean": f"{float(np.mean(blend_pixels)):.6f}" if blend_pixels else "0",
            }
        )
        print(f"[{pos}/{len(rows)}] {output_name} | {status} frames={written}/{target_n} flow={len(flow_frames)}", flush=True)

    with (args.out_root / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    with (args.out_root / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diag[0].keys()))
        writer.writeheader()
        writer.writerows(diag)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
