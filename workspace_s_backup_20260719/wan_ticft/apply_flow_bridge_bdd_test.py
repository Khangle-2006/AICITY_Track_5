"""Apply early optical-flow bridge to BDD test samples in a submission folder."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"cannot write {path}")


def read_bdd_rows(test_csv: Path) -> list[dict[str, str]]:
    with test_csv.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("view_type") == "bdd"]


def resize_like(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == ref.shape[:2]:
        return frame
    return cv2.resize(frame, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32), 0, 255).astype(np.uint8)


def build_frames(
    history: list[np.ndarray],
    base: list[np.ndarray],
    bridge_frames: int,
    blend_frames: int,
    flow_width: int,
) -> list[np.ndarray]:
    if len(history) < 2:
        return base
    offsets = list(range(1, len(base) + 1))
    flow_frames = region_predictions(cv2, np, history[-2], history[-1], offsets, 1, flow_width)
    flow_frames = [resize_like(frame, base[0]) for frame in flow_frames]
    out = []
    for idx, base_frame in enumerate(base):
        if idx < bridge_frames:
            out.append(flow_frames[idx])
        elif blend_frames > 0 and idx < bridge_frames + blend_frames:
            alpha = (idx - bridge_frames + 1) / float(blend_frames + 1)
            out.append(blend(flow_frames[idx], base_frame, alpha))
        else:
            out.append(base_frame)
    return out


def process_folder(args: argparse.Namespace, out_dir: Path, bridge_frames: int, blend_frames: int) -> None:
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_bdd_rows(args.test_csv)
    report = []
    for index, row in enumerate(rows, 1):
        sample = row["source_folder"]
        expected = int(row["target_N"])
        dst = out_dir / sample
        if len(numeric_pngs(dst)) == expected:
            print(f"{out_dir.name} [{index}/{len(rows)}] {sample} resume", flush=True)
            continue
        input_paths = numeric_pngs(Path(row["input_dir"]))
        base_paths = numeric_pngs(args.base_dir / sample)
        if len(base_paths) != expected:
            raise RuntimeError(f"{sample}: expected {expected} base frames, got {len(base_paths)}")
        history = [read_rgb(path) for path in input_paths[-2:]]
        base = [read_rgb(path) for path in base_paths]
        frames = build_frames(history, base, bridge_frames, blend_frames, args.flow_width)
        if dst.exists():
            shutil.rmtree(dst)
        for idx, frame in enumerate(frames):
            write_rgb(dst / f"{idx}.png", frame)
        verified = verify_sample_output(dst, expected)
        if not verified.ok:
            raise RuntimeError(f"{sample}: verify failed {verified.notes}")
        report.append({
            "sample_id": row["sample_id"],
            "output_name": sample,
            "target_N": expected,
            "frames_found": verified.frames_found,
            "bridge_frames": bridge_frames,
            "blend_frames": blend_frames,
            "status": "ok",
        })
        print(f"{out_dir.name} [{index}/{len(rows)}] {sample} frames={expected}", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "output_name", "target_N", "frames_found", "bridge_frames", "blend_frames", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="name=bridge,blend")
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    for value in args.variant:
        name, spec = value.split("=", 1)
        bridge, blend_count = [int(part) for part in spec.split(",")]
        process_folder(args, args.out_root / name, bridge, blend_count)


if __name__ == "__main__":
    main()
