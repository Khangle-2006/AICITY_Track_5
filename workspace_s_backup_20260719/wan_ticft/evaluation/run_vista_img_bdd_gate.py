from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


def read_rows(path: Path, limit: int, view_type: str, sample_ids: list[str] | None) -> list[dict]:
    rows: list[dict] = []
    wanted = set(sample_ids or [])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if row.get("view_type") != view_type:
                continue
            if wanted and sample_id not in wanted:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_video_bgr(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def crop_resize_bgr(cv2, frame, width: int, height: int):
    h, w = frame.shape[:2]
    target_ratio = width / height
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        left = max(0, (w - new_w) // 2)
        frame = frame[:, left : left + new_w]
    elif ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        top = max(0, (h - new_h) // 2)
        frame = frame[top : top + new_h, :]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def materialize_future_video(
    cv2,
    vista_video: Path,
    out_path: Path,
    anchor_rgb,
    target_count: int,
    source_fps: float,
    out_width: int,
    out_height: int,
) -> int:
    frames, vista_fps = read_video_bgr(cv2, vista_video)
    # Vista outputs include the conditioning image as the first frame. Drop it
    # so frame 0 in the report is the first future frame.
    future = frames[1:] if len(frames) > 1 else frames
    anchor_bgr = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR)
    rendered = [crop_resize_bgr(cv2, anchor_bgr, out_width, out_height)]
    for target_index in range(target_count):
        seconds = target_index / max(source_fps, 1.0)
        vista_index = min(len(future) - 1, max(0, int(round(seconds * vista_fps))))
        rendered.append(crop_resize_bgr(cv2, future[vista_index], out_width, out_height))
    write_video(cv2, rendered, out_path, source_fps)
    return len(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vista_repo", type=Path, default=Path("/workspace/third_party/Vista"))
    parser.add_argument("--vista_python", type=Path, default=Path("/workspace/experiments/wan_ticft/vista_env/bin/python"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument("--n_rounds", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=25)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--out_height", type=int, default=720)
    parser.add_argument("--out_width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse_raw", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = read_rows(args.manifest, args.limit, args.view_type, args.sample_ids)
    if len(rows) < args.limit:
        raise RuntimeError(f"found only {len(rows)} rows for view_type={args.view_type}")
    if not args.vista_repo.exists():
        raise RuntimeError(f"missing Vista repo: {args.vista_repo}")
    if not args.vista_python.exists():
        raise RuntimeError(f"missing Vista python: {args.vista_python}")
    ckpt = args.vista_repo / "ckpts" / "vista.safetensors"
    if not ckpt.exists():
        raise RuntimeError(f"missing Vista checkpoint: {ckpt}")

    if args.out_dir.exists() and not args.resume:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    completed_rows = []
    image_folder = args.vista_repo / "image_folder"

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        future_path = args.out_dir / "videos" / f"{sample_id}.mp4"
        target_paths = (row.get("frames") or {}).get("target") or []
        cond_paths = (row.get("frames") or {}).get("condition") or []
        target_count = len(target_paths)
        if not target_count or not cond_paths:
            raise RuntimeError(f"{sample_id}: missing condition or target frames")
        source_fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        raw_dir = args.out_dir / "raw" / sample_id
        raw_videos = list((raw_dir / "virtual" / "videos").glob("*.mp4"))
        if args.reuse_raw and raw_videos:
            last_condition = read_rgb(cv2, Path(cond_paths[-1]))
            future_frames = materialize_future_video(
                cv2,
                raw_videos[0],
                future_path,
                last_condition,
                target_count,
                source_fps,
                args.out_width,
                args.out_height,
            )
            report.append(
                {
                    "sample_id": sample_id,
                    "video": str(future_path),
                    "status": "ok",
                    "source_fps": source_fps,
                    "future_frames": future_frames,
                    "notes": f"IMG last-frame anchor; rematerialized raw={raw_videos[0]}",
                }
            )
            completed_rows.append(row)
            continue
        if args.resume and future_path.exists():
            report.append(
                {
                    "sample_id": sample_id,
                    "video": str(future_path),
                    "status": "ok",
                    "source_fps": source_fps,
                    "future_frames": target_count,
                    "notes": "resume",
                }
            )
            completed_rows.append(row)
            continue

        image_folder.mkdir(parents=True, exist_ok=True)
        for old in image_folder.glob("*"):
            if old.is_file():
                old.unlink()
        last_condition = read_rgb(cv2, Path(cond_paths[-1]))
        input_png = image_folder / f"{sample_id}.png"
        if not cv2.imwrite(str(input_png), cv2.cvtColor(last_condition, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"{sample_id}: cannot write {input_png}")

        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        command = [
            str(args.vista_python),
            "sample.py",
            "--dataset",
            "IMG",
            "--save",
            str(raw_dir),
            "--n_rounds",
            str(args.n_rounds),
            "--n_frames",
            str(args.n_frames),
            "--n_conds",
            "1",
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--n_steps",
            str(args.n_steps),
            "--seed",
            str(args.seed),
            "--rand_gen",
        ]
        if args.low_vram:
            command.append("--low_vram")
        print(f"[{position}/{len(rows)}] Vista IMG {sample_id}", flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.vista_repo)
        env["TOKENIZERS_PARALLELISM"] = "false"
        # Vista uses torch 2.0.1; newer allocator knobs such as
        # expandable_segments are rejected at CUDA initialization.
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        subprocess.run(command, cwd=args.vista_repo, env=env, check=True)
        raw_videos = list((raw_dir / "virtual" / "videos").glob("*.mp4"))
        if len(raw_videos) != 1:
            raise RuntimeError(f"{sample_id}: expected one Vista video, found {len(raw_videos)}")
        future_frames = materialize_future_video(
            cv2,
            raw_videos[0],
            future_path,
            last_condition,
            target_count,
            source_fps,
            args.out_width,
            args.out_height,
        )
        report.append(
            {
                "sample_id": sample_id,
                "video": str(future_path),
                "status": "ok",
                "source_fps": source_fps,
                "future_frames": future_frames,
                "notes": f"IMG last-frame anchor; raw={raw_videos[0]}",
            }
        )
        completed_rows.append(row)

    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in completed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(report), "requested": args.limit, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
