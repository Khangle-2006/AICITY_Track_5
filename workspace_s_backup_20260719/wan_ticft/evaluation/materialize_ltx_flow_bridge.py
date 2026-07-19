from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f) if row.get("status") == "ok"}


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def read_video_rgb(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, fps


def resize_like(cv2, frame, reference):
    if frame.shape[:2] == reference.shape[:2]:
        return frame
    h, w = reference.shape[:2]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)


def condition_gap(row: dict) -> int:
    indices = (row.get("frame_indices") or {}).get("condition") or []
    if len(indices) >= 2:
        return max(1, int(indices[-1]) - int(indices[-2]))
    return 1


def parse_variant(value: str) -> tuple[str, int, int]:
    if "=" not in value:
        raise ValueError(f"variant must be label=bridge,blend, got {value!r}")
    label, spec = value.split("=", 1)
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"variant spec must be bridge,blend, got {spec!r}")
    return label.strip(), int(parts[0]), int(parts[1])


def blend_frames(np, a, b, alpha: float):
    return np.clip((1.0 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32), 0, 255).astype(
        np.uint8
    )


def build_bridge_frames(cv2, np, row: dict, ltx_frames, bridge_frames: int, blend_count: int, width: int):
    from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions

    condition_paths = ((row.get("frames") or {}).get("condition") or [])
    conditions = [img for path in condition_paths if (img := read_rgb(cv2, Path(path))) is not None]
    if len(conditions) < 2:
        raise RuntimeError(f"{row.get('sample_id')}: need at least two condition frames")

    offsets = list(range(1, len(ltx_frames) + 1))
    bridge = region_predictions(cv2, np, conditions[-2], conditions[-1], offsets, condition_gap(row), width)
    bridge = [resize_like(cv2, frame, ltx_frames[0]) for frame in bridge]

    out = []
    for index, ltx in enumerate(ltx_frames):
        if index < bridge_frames:
            out.append(bridge[index])
            continue
        if blend_count > 0 and index < bridge_frames + blend_count:
            alpha = (index - bridge_frames + 1) / float(blend_count + 1)
            out.append(blend_frames(np, bridge[index], ltx, alpha))
            continue
        out.append(ltx)
    return out


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "video",
        "status",
        "source_fps",
        "future_frames",
        "bridge_frames",
        "blend_frames",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ltx_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="label=bridge_frames,blend_frames")
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = read_manifest(args.manifest)
    reports = read_report(args.ltx_report)
    variants = [parse_variant(value) for value in args.variant]

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for label, bridge_frames, blend_frames_count in variants:
        out_dir = args.out_root / label
        report_rows = []
        for position, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]
            source = reports.get(sample_id)
            if source is None:
                print(f"{label} [{position}/{len(rows)}] {sample_id} | skipped", flush=True)
                continue
            ltx_frames, fps = read_video_rgb(cv2, Path(source["video"]))
            frames = build_bridge_frames(
                cv2,
                np,
                row,
                ltx_frames,
                bridge_frames=bridge_frames,
                blend_count=blend_frames_count,
                width=args.flow_width,
            )
            video_path = out_dir / "videos" / f"{sample_id}.mp4"
            write_video(cv2, frames, video_path, fps)
            report_rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "status": "ok",
                    "source_fps": source.get("source_fps", ""),
                    "future_frames": len(frames),
                    "bridge_frames": bridge_frames,
                    "blend_frames": blend_frames_count,
                    "notes": "early region-flow bridge into raw LTX",
                }
            )
            print(f"{label} [{position}/{len(rows)}] {sample_id} | ok", flush=True)
        write_report(out_dir / "report.csv", report_rows)


if __name__ == "__main__":
    main()
