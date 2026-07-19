from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb


def read_rows(path: Path, limit: int, sample_ids: list[str] | None) -> list[dict[str, Any]]:
    wanted = set(sample_ids or [])
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if row.get("view_type") != "bdd":
                continue
            if wanted and sample_id not in wanted:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def read_history(cv2, video_path: Path, last_index: int, count: int):
    start = max(0, last_index - count + 1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    for index in range(start, last_index + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot read {video_path} at frame {index}")
        frames.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return frames, fps


def align_history_to_reference(cv2, frames, reference):
    if reference is None:
        return frames, "unchecked", 0.0

    candidates = {
        "none": frames,
        "clockwise": [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in frames],
        "counterclockwise": [cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) for frame in frames],
    }
    scored = []
    for name, candidate in candidates.items():
        last_rgb = cv2.cvtColor(candidate[-1], cv2.COLOR_BGR2RGB)
        if last_rgb.shape[:2] != reference.shape[:2]:
            last_rgb = cv2.resize(last_rgb, (reference.shape[1], reference.shape[0]))
        error = float(abs(last_rgb.astype("float32") - reference.astype("float32")).mean())
        scored.append((error, name, candidate))
    error, name, aligned = min(scored, key=lambda item: item[0])
    return aligned, name, error


def force_orientation(cv2, frames, mode: str):
    if mode == "manifest":
        return frames, "manifest"
    out = []
    changed = False
    for frame in frames:
        h, w = frame.shape[:2]
        new_frame = frame
        if mode == "landscape_cw" and h > w:
            new_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            changed = True
        elif mode == "landscape_ccw" and h > w:
            new_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            changed = True
        elif mode == "portrait_cw" and w > h:
            new_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            changed = True
        elif mode == "portrait_ccw" and w > h:
            new_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            changed = True
        out.append(new_frame)
    return out, mode if changed else f"{mode}_unchanged"


def resize_frames(cv2, frames, width: int, height: int):
    out = []
    for frame in frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.append(frame)
    return out


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def cosmos_prompt(row: dict[str, Any]) -> str:
    base = str(row.get("prompt", "")).strip()
    if base:
        base = " ".join(base.split()[:90])
    return " ".join(
        [
            "Continue this forward-facing driving video from the same camera.",
            "Preserve the exact camera orientation, road layout, lighting, windshield view, vehicles, pedestrians, and object identities.",
            "Keep the horizon level and continue realistic forward motion with sharp geometry.",
            "Do not rotate the camera, change the scene, add new actors, or remove visible actors.",
            base,
        ]
    ).strip()


def run_json_command(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed {proc.returncode}: {' '.join(command)}\n{proc.stderr}\n{proc.stdout}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"expected JSON from command:\n{proc.stdout}\n{proc.stderr}") from exc


def submit_job(args, video_path: Path, row: dict[str, Any], seed: int) -> str:
    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": True,
        "condition_frame_indexes_vision": args.condition_frame_indexes,
        "condition_video_keep": args.condition_video_keep,
    }
    command = [
        "curl",
        "-sS",
        "-X",
        "POST",
        f"{args.server}/v1/videos",
        "-H",
        "Accept: application/json",
        "-F",
        f"prompt={cosmos_prompt(row)}",
        "-F",
        "negative_prompt=tilted camera, rotated horizon, portrait video, scene change, blur, ghosting, double exposure, warped geometry, hallucinated objects, low quality, jittery, deformed",
        "-F",
        f"size={args.width}x{args.height}",
        "-F",
        f"num_frames={args.num_frames}",
        "-F",
        f"fps={args.fps}",
        "-F",
        f"num_inference_steps={args.steps}",
        "-F",
        f"guidance_scale={args.guidance_scale}",
        "-F",
        f"flow_shift={args.flow_shift}",
        "-F",
        f"seed={seed}",
        "-F",
        f"extra_params={json.dumps(extra, separators=(',', ':'))}",
        "-F",
        f"input_reference=@{video_path};type=video/mp4",
    ]
    response = run_json_command(command)
    if response.get("error"):
        raise RuntimeError(json.dumps(response, indent=2))
    job_id = str(response.get("id") or "")
    if not job_id:
        raise RuntimeError(f"missing job id: {json.dumps(response, indent=2)}")
    return job_id


def submit_image_job(args, image_path: Path, row: dict[str, Any], seed: int) -> str:
    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": True,
    }
    prompt = " ".join(
        [
            "Use the reference image as a strict first-frame anchor.",
            "Continue the exact same forward-facing driving scene.",
            "Do not change the road, buildings, cars, pedestrians, lane markings, windshield, dashboard, camera angle, or horizon.",
            "Only advance existing objects slightly in time.",
            "No new scene, no new location, no new people, no new vehicles.",
            cosmos_prompt(row),
        ]
    )
    command = [
        "curl",
        "-sS",
        "-X",
        "POST",
        f"{args.server}/v1/videos",
        "-H",
        "Accept: application/json",
        "-F",
        f"prompt={prompt}",
        "-F",
        "negative_prompt=new scene, different street, different city, changed buildings, changed cars, changed pedestrians, tilted camera, rotated horizon, portrait video, scene change, blur, ghosting, double exposure, warped geometry, hallucinated objects, low quality, jittery, deformed",
        "-F",
        f"size={args.width}x{args.height}",
        "-F",
        f"num_frames={args.num_frames}",
        "-F",
        f"fps={args.fps}",
        "-F",
        f"num_inference_steps={args.steps}",
        "-F",
        f"guidance_scale={args.guidance_scale}",
        "-F",
        f"flow_shift={args.flow_shift}",
        "-F",
        f"seed={seed}",
        "-F",
        f"extra_params={json.dumps(extra, separators=(',', ':'))}",
        "-F",
        f"input_reference=@{image_path};type=image/png",
    ]
    response = run_json_command(command)
    if response.get("error"):
        raise RuntimeError(json.dumps(response, indent=2))
    job_id = str(response.get("id") or "")
    if not job_id:
        raise RuntimeError(f"missing job id: {json.dumps(response, indent=2)}")
    return job_id


def poll_job(args, job_id: str) -> dict[str, Any]:
    deadline = time.time() + args.timeout
    last = {}
    while time.time() < deadline:
        last = run_json_command(["curl", "-sS", f"{args.server}/v1/videos/{job_id}"])
        status = last.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return last
        print(f"{job_id}: status={status} progress={last.get('progress')}", flush=True)
        time.sleep(args.poll_interval)
    raise TimeoutError(f"{job_id}: timed out; last={json.dumps(last, indent=2)}")


def download_video(args, job_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-sS",
        "-fL",
        f"{args.server}/v1/videos/{job_id}/content",
        "-o",
        str(out_path),
    ]
    subprocess.run(command, check=True)
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"{job_id}: bad download {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl"))
    parser.add_argument("--out_dir", type=Path, default=Path("/workspace/experiments/wan_ticft/cosmos3_bdd_v2v_smoke"))
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument("--history_frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--orientation", choices=["manifest", "landscape_cw", "landscape_ccw", "portrait_cw", "portrait_ccw"], default="landscape_cw")
    parser.add_argument("--condition_video_keep", choices=["first", "last"], default="last")
    parser.add_argument("--condition_frame_indexes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--conditioning_mode", choices=["video", "image"], default="video")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll_interval", type=float, default=5.0)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = read_rows(args.manifest, args.limit, args.sample_ids)
    if len(rows) != args.limit:
        raise RuntimeError(f"found {len(rows)} rows, expected {args.limit}")

    if args.out_dir.exists() and not args.resume:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    selected_manifest = []
    for pos, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        out_video = args.out_dir / "videos" / f"{sample_id}.mp4"
        history_video = args.out_dir / "history_videos" / f"{sample_id}.mp4"
        anchor_image = args.out_dir / "anchor_images" / f"{sample_id}.png"
        if args.resume and out_video.exists():
            print(f"[{pos}/{len(rows)}] {sample_id} | resume", flush=True)
            report.append({"sample_id": sample_id, "video": str(out_video), "status": "ok", "job_id": "", "notes": "resume"})
            selected_manifest.append(row)
            continue

        condition = (row.get("frame_indices") or {}).get("condition") or []
        if not condition:
            raise RuntimeError(f"{sample_id}: missing condition indices")
        last_index = int(condition[-1])
        frames, source_fps = read_history(cv2, Path(row["video_path"]), last_index, args.history_frames)
        condition_paths = (row.get("frames") or {}).get("condition") or []
        reference = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
        frames, align_name, align_error = align_history_to_reference(cv2, frames, reference)
        frames, orientation_name = force_orientation(cv2, frames, args.orientation)
        frames = resize_frames(cv2, frames, args.width, args.height)
        write_video(cv2, frames, history_video, source_fps)
        anchor_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(anchor_image), frames[-1]):
            raise RuntimeError(f"{sample_id}: cannot write {anchor_image}")

        print(
            f"[{pos}/{len(rows)}] {sample_id} | align={align_name} err={align_error:.3f} "
            f"orientation={orientation_name} history={history_video}",
            flush=True,
        )
        if args.conditioning_mode == "image":
            job_id = submit_image_job(args, anchor_image, row, args.seed + pos - 1)
        else:
            job_id = submit_job(args, history_video, row, args.seed + pos - 1)
        status = poll_job(args, job_id)
        if status.get("status") != "completed":
            raise RuntimeError(f"{sample_id}: job failed: {json.dumps(status, indent=2)}")
        download_video(args, job_id, out_video)
        report.append(
            {
                "sample_id": sample_id,
                "video": str(out_video),
                "status": "ok",
                "job_id": job_id,
                "source_fps": f"{source_fps:.6f}",
                "history_frames": len(frames),
                "future_frames": args.num_frames,
                "orientation": orientation_name,
                "align": align_name,
                "align_error": f"{align_error:.6f}",
                "conditioning_mode": args.conditioning_mode,
                "inference_time_s": status.get("inference_time_s", ""),
                "peak_memory_mb": status.get("peak_memory_mb", ""),
            }
        )
        selected_manifest.append(row)

    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(report), "out_dir": str(args.out_dir), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
