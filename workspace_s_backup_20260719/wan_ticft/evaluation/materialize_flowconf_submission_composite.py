"""Composite flow-confidence frames into a submission-style PNG frame tree."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_base_report(path: Path) -> dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                out[row["sample_id"]] = row
    return out


def read_flow_report(path: Path) -> dict[str, Path]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                out[row["sample_id"]] = Path(row["video"])
    return out


def read_video_frames(cv2, video: Path, count: int):
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
    gray = cv2.imread(str(alpha_dir / f"alpha_{index:03d}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(reference.shape[:2], dtype=np.float32)
    if gray.shape[:2] != reference.shape[:2]:
        gray = cv2.resize(gray, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = gray.astype(np.float32) / 255.0
    mask = np.clip((alpha - threshold) / max(softness, 1e-6), 0.0, 1.0)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_root", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--flow_report", type=Path, required=True)
    parser.add_argument("--alpha_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--softness", type=float, default=0.08)
    parser.add_argument("--blur", type=float, default=2.0)
    parser.add_argument("--phase_include", nargs="*", default=None)
    parser.add_argument("--copy_unselected", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    base_report = read_base_report(args.base_report)
    flow_report = read_flow_report(args.flow_report)
    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type]
    phase_include = set(str(item) for item in args.phase_include) if args.phase_include is not None else None
    report_rows = []
    for position, row in enumerate(rows, start=1):
        sid = row["sample_id"]
        base_meta = base_report.get(sid)
        flow_video = flow_report.get(sid)
        if base_meta is None or flow_video is None:
            print(f"[{position}/{len(rows)}] {sid} | skip", flush=True)
            continue
        phase = str(row.get("phase", ""))
        use_composite = phase_include is None or phase in phase_include
        output_name = base_meta["output_name"]
        count = int(base_meta.get("target_N") or base_meta.get("frames_found") or 0)
        base_dir = args.base_root / output_name
        out_dir = args.out_root / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not use_composite:
            written = 0
            if args.copy_unselected:
                for index in range(count):
                    source = base_dir / f"{index}.png"
                    if source.exists():
                        shutil.copy2(source, out_dir / f"{index}.png")
                        written += 1
            report_rows.append(
                {
                    "output_name": output_name,
                    "sample_id": sid,
                    "view_type": row.get("view_type", ""),
                    "target_N": count,
                    "frames_found": written,
                    "status": "ok" if written == count else "bad",
                }
            )
            print(f"[{position}/{len(rows)}] {sid} | copied {written}/{count}", flush=True)
            continue
        flow_frames = read_video_frames(cv2, flow_video, count)
        written = 0
        for index in range(count):
            base_bgr = cv2.imread(str(base_dir / f"{index}.png"), cv2.IMREAD_COLOR)
            if base_bgr is None or index >= len(flow_frames):
                continue
            base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
            flow = resize_like(cv2, flow_frames[index], base)
            mask = load_alpha(cv2, np, args.alpha_root / sid, index, base, args.threshold, args.softness, args.blur)
            weight = np.clip(mask * args.strength, 0.0, 1.0)[:, :, None]
            frame = np.clip(base.astype(np.float32) * (1.0 - weight) + flow.astype(np.float32) * weight, 0, 255).astype(np.uint8)
            ok = cv2.imwrite(str(out_dir / f"{index}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += int(bool(ok))
        report_rows.append(
            {
                "output_name": output_name,
                "sample_id": sid,
                "view_type": row.get("view_type", ""),
                "target_N": count,
                "frames_found": written,
                "status": "ok" if written == count else "bad",
            }
        )
        print(f"[{position}/{len(rows)}] {sid} | {written}/{count}", flush=True)

    with (args.out_root / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output_name", "sample_id", "view_type", "target_N", "frames_found", "status"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
