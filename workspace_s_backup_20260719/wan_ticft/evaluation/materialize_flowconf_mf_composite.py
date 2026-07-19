"""Composite learned flow-confidence pixels into an MF base video report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_targets(cv2, row: dict):
    return [image for path in (row.get("frames") or {}).get("target", []) if (image := read_rgb(cv2, Path(path))) is not None]


def read_video_sequence(cv2, video: Path, count: int):
    cap = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < count:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def resize_like(cv2, image, reference):
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def load_alpha(cv2, np, alpha_dir: Path, index: int, reference, threshold: float, softness: float, blur: float):
    path = alpha_dir / f"alpha_{index:03d}.png"
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(reference.shape[:2], dtype=np.float32)
    if gray.shape[:2] != reference.shape[:2]:
        gray = cv2.resize(gray, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = gray.astype(np.float32) / 255.0
    mask = (alpha - threshold) / max(softness, 1e-6)
    mask = np.clip(mask, 0.0, 1.0)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def write_video(cv2, frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in [frames[0], *frames]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def score(cv2, np, predictions, targets):
    values = [(psnr(cv2, np, p, t), ssim_gray(cv2, np, p, t), mean_absdiff(np, p, t)) for p, t in zip(predictions, targets)]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--flow_report", type=Path, required=True)
    parser.add_argument("--alpha_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.04, 0.07, 0.10])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.35, 0.65, 1.0])
    parser.add_argument("--softness", type=float, default=0.08)
    parser.add_argument("--blur", type=float, default=2.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type][: args.limit]
    base_report = read_report(args.base_report)
    flow_report = read_report(args.flow_report)

    variants = [(threshold, strength) for threshold in args.thresholds for strength in args.strengths]
    report_rows = {variant: [] for variant in variants}
    eval_rows = []

    for position, row in enumerate(rows, start=1):
        sid = row["sample_id"]
        base_video = base_report.get(sid)
        flow_video = flow_report.get(sid)
        targets = load_targets(cv2, row)
        if base_video is None or flow_video is None:
            print(f"[{position}/{len(rows)}] {sid} | skip", flush=True)
            continue
        if targets:
            references = targets
            offsets = map_target_offsets(row, video_fps(cv2, base_video))
        else:
            target_count = int((row.get("ticft") or {}).get("target_frames") or 9)
            references = [read_rgb(cv2, Path(((row.get("frames") or {}).get("condition") or [])[-1]))] * target_count
            offsets = list(range(1, target_count + 1))
        if not references or any(item is None for item in references):
            print(f"[{position}/{len(rows)}] {sid} | skip(no references)", flush=True)
            continue
        base_map = read_video_rgb_frames(cv2, base_video, offsets)
        base_frames = [resize_like(cv2, base_map[index], target) for index, target in zip(offsets, references) if index in base_map]
        flow_frames = [resize_like(cv2, frame, target) for frame, target in zip(read_video_sequence(cv2, flow_video, len(references)), references)]
        if len(base_frames) != len(references) or len(flow_frames) != len(references):
            print(f"[{position}/{len(rows)}] {sid} | skip frame count base={len(base_frames)} flow={len(flow_frames)} target={len(references)}", flush=True)
            continue
        if targets:
            p, s, d = score(cv2, np, base_frames, targets)
            eval_rows.append({"sample_id": sid, "candidate": "mf_base", "threshold": "", "strength": "", "psnr": f"{p:.6f}", "ssim": f"{s:.6f}", "absdiff": f"{d:.6f}"})

        alpha_dir = args.alpha_root / sid
        for threshold, strength in variants:
            output = []
            for index, (base, flow) in enumerate(zip(base_frames, flow_frames)):
                mask = load_alpha(cv2, np, alpha_dir, index, base, threshold, args.softness, args.blur)
                weight = np.clip(mask * strength, 0.0, 1.0)[:, :, None]
                frame = np.clip(base.astype(np.float32) * (1.0 - weight) + flow.astype(np.float32) * weight, 0, 255).astype(np.uint8)
                output.append(frame)
            label = f"flowconf_t{int(round(threshold * 100)):02d}_s{int(round(strength * 100)):03d}"
            video = args.out_dir / label / "videos" / f"{sid}.avi"
            write_video(cv2, output, video, video_fps(cv2, base_video) or 30.0)
            report_rows[(threshold, strength)].append({"sample_id": sid, "video": str(video), "status": "ok", "future_frames": len(output)})
            if targets:
                p, s, d = score(cv2, np, output, targets)
                eval_rows.append({"sample_id": sid, "candidate": label, "threshold": threshold, "strength": strength, "psnr": f"{p:.6f}", "ssim": f"{s:.6f}", "absdiff": f"{d:.6f}"})
        print(f"[{position}/{len(rows)}] {sid} | ok", flush=True)

    for threshold, strength in variants:
        label = f"flowconf_t{int(round(threshold * 100)):02d}_s{int(round(strength * 100)):03d}"
        out = args.out_dir / label / "report.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
            writer.writeheader()
            writer.writerows(report_rows[(threshold, strength)])

    if eval_rows:
        with (args.out_dir / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=eval_rows[0].keys())
            writer.writeheader()
            writer.writerows(eval_rows)

        print()
        print(f"{'Candidate':<22} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
        print("-" * 66)
        for candidate in sorted({row["candidate"] for row in eval_rows}):
            items = [row for row in eval_rows if row["candidate"] == candidate]
            print(f"{candidate:<22} {len(items):>5} {np.mean([float(x['psnr']) for x in items]):>10.4f} {np.mean([float(x['ssim']) for x in items]):>10.4f} {np.mean([float(x['absdiff']) for x in items]):>10.4f}")
    print(f"output {args.out_dir}")


if __name__ == "__main__":
    main()
