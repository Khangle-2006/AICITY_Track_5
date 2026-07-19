from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_rows(manifest: Path, limit: int, view_type: str, sample_ids: list[str] | None) -> list[dict]:
    wanted = set(sample_ids or [])
    rows = []
    for row in read_jsonl(manifest):
        if row.get("view_type") != view_type:
            continue
        if wanted and row.get("sample_id") not in wanted:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def write_video(cv2, frames_rgb, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def prepare_lr_sequences(
    cv2,
    manifest_rows: list[dict],
    base_report: Path,
    in_dir: Path,
    lr_width: int,
    lr_height: int,
) -> None:
    report = read_report(base_report)
    if in_dir.exists():
        shutil.rmtree(in_dir)
    in_dir.mkdir(parents=True)
    for row in manifest_rows:
        sample_id = row["sample_id"]
        video = report.get(sample_id)
        if video is None:
            raise RuntimeError(f"{sample_id}: missing from base report {base_report}")
        offsets = map_target_offsets(row, video_fps(cv2, video))
        frame_map = read_video_rgb_frames(cv2, video, offsets)
        seq_dir = in_dir / sample_id
        seq_dir.mkdir(parents=True)
        for pos, offset in enumerate(offsets):
            frame = frame_map.get(offset)
            if frame is None:
                raise RuntimeError(f"{sample_id}: missing base frame at offset {offset}")
            small = cv2.resize(frame, (lr_width, lr_height), interpolation=cv2.INTER_AREA)
            out = seq_dir / f"{pos:08d}.png"
            if not cv2.imwrite(str(out), cv2.cvtColor(small, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"{sample_id}: cannot write {out}")


def sr_sequence_dir(sr_out_dir: Path, sample_id: str) -> Path:
    direct = sr_out_dir / sample_id
    if direct.exists():
        return direct
    nested = list(sr_out_dir.glob(f"*/{sample_id}"))
    return nested[0] if nested else direct


def has_complete_sr_output(sr_out_dir: Path, manifest_rows: list[dict]) -> bool:
    if not sr_out_dir.exists():
        return False
    for row in manifest_rows:
        sample_id = row["sample_id"]
        targets = (row.get("frames") or {}).get("target") or []
        seq_dir = sr_sequence_dir(sr_out_dir, sample_id)
        frame_paths = sorted(seq_dir.glob("*.png")) if seq_dir.exists() else []
        if len(frame_paths) < len(targets):
            return False
    return True


def materialize_report_videos(
    cv2,
    manifest_rows: list[dict],
    sr_out_dir: Path,
    out_video_dir: Path,
    report_path: Path,
    out_width: int,
    out_height: int,
) -> None:
    out_video_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        seq_dir = sr_sequence_dir(sr_out_dir, sample_id)
        frame_paths = sorted(seq_dir.glob("*.png"))
        targets = (row.get("frames") or {}).get("target") or []
        if len(frame_paths) < len(targets):
            raise RuntimeError(f"{sample_id}: StableVSR wrote {len(frame_paths)}/{len(targets)} frames")
        anchor = read_rgb(cv2, Path(((row.get("frames") or {}).get("condition") or [])[-1]))
        if anchor is None:
            raise RuntimeError(f"{sample_id}: cannot read last condition")
        frames = [cv2.resize(anchor, (out_width, out_height), interpolation=cv2.INTER_AREA)]
        for path in frame_paths[: len(targets)]:
            frame = read_rgb(cv2, path)
            if frame is None:
                raise RuntimeError(f"{sample_id}: cannot read {path}")
            frames.append(cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA))
        fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        video = out_video_dir / f"{sample_id}.mp4"
        write_video(cv2, frames, video, fps)
        rows.append(
            {
                "sample_id": sample_id,
                "video": str(video),
                "status": "ok",
                "source_fps": fps,
                "future_frames": len(targets),
                "notes": "depth_a75 downscale -> StableVSR x4; frame0 is anchor",
            }
        )
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--stablevsr_repo", type=Path, default=Path("/workspace/third_party/StableVSR"))
    parser.add_argument("--stablevsr_python", type=Path, default=Path("/workspace/experiments/wan_ticft/stablevsr_env/bin/python"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--lr_width", type=int, default=320)
    parser.add_argument("--lr_height", type=int, default=180)
    parser.add_argument("--out_width", type=int, default=1280)
    parser.add_argument("--out_height", type=int, default=720)
    parser.add_argument("--reuse_sr", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = selected_rows(args.manifest, args.limit, args.view_type, args.sample_ids)
    if len(rows) < args.limit:
        raise RuntimeError(f"found only {len(rows)} rows")
    if not args.stablevsr_repo.exists():
        raise RuntimeError(f"missing StableVSR repo: {args.stablevsr_repo}")
    if not args.stablevsr_python.exists():
        raise RuntimeError(f"missing StableVSR python: {args.stablevsr_python}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = args.out_dir / "stablevsr_lr_input"
    sr_out_dir = args.out_dir / "stablevsr_output"
    videos_dir = args.out_dir / "videos"
    prepare_lr_sequences(cv2, rows, args.base_report, in_dir, args.lr_width, args.lr_height)

    if not args.reuse_sr or not has_complete_sr_output(sr_out_dir, rows):
        if sr_out_dir.exists():
            shutil.rmtree(sr_out_dir)
        command = [
            str(args.stablevsr_python),
            "test.py",
            "--in_path",
            str(in_dir),
            "--out_path",
            str(sr_out_dir),
            "--num_inference_steps",
            str(args.num_inference_steps),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.stablevsr_repo)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        subprocess.run(command, cwd=args.stablevsr_repo, env=env, check=True)

    materialize_report_videos(
        cv2,
        rows,
        sr_out_dir,
        videos_dir,
        args.out_dir / "report.csv",
        args.out_width,
        args.out_height,
    )
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "status": "ok", "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
