"""Run a small BDD Seedance gate through fal.ai and write a video report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb


DEFAULT_PROMPT = (
    "@Video1 shows the observed driving scene. Continue the same traffic video "
    "for the near future. Preserve the exact road layout, camera viewpoint, "
    "lighting, ego motion, lane geometry, vehicles, pedestrians, and background. "
    "Do not add new cars or people. Keep motion natural and physically plausible."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_history_clip(cv2, row: dict[str, Any], path: Path, width: int = 854, height: int = 480) -> None:
    condition_paths = (row.get("frames") or {}).get("condition") or []
    frames = [image for item in condition_paths if (image := read_rgb(cv2, Path(item))) is not None]
    if len(frames) < 2:
        raise RuntimeError(f"{row.get('sample_id')}: need at least 2 condition frames, got {len(frames)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1.0, min(4.0, len(frames) / 2.0))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    writer.release()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        path.write_bytes(response.read())


def run_fal_seedance2_ref(
    fal_client,
    history_path: Path,
    prompt: str,
    resolution: str,
    duration: str,
    seed: int,
    fast: bool,
) -> dict[str, Any]:
    model = (
        "bytedance/seedance-2.0/fast/reference-to-video"
        if fast
        else "bytedance/seedance-2.0/reference-to-video"
    )
    video_url = fal_client.upload_file(str(history_path))
    return fal_client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "video_urls": [video_url],
            "resolution": resolution,
            "duration": duration,
            "aspect_ratio": "16:9",
            "generate_audio": False,
            "seed": seed,
        },
        with_logs=True,
    )


def run_fal_seedance1_i2v(
    fal_client,
    image_path: Path,
    prompt: str,
    resolution: str,
    duration: str,
    seed: int,
) -> dict[str, Any]:
    image_url = fal_client.upload_file(str(image_path))
    return fal_client.subscribe(
        "fal-ai/bytedance/seedance/v1/pro/image-to-video",
        arguments={
            "prompt": prompt.replace("@Video1", "The image"),
            "image_url": image_url,
            "resolution": resolution,
            "duration": duration,
            "aspect_ratio": "16:9",
            "camera_fixed": False,
            "seed": seed,
            "enable_safety_checker": True,
        },
        with_logs=True,
    )


def write_last_condition_image(cv2, row: dict[str, Any], path: Path) -> None:
    condition_paths = (row.get("frames") or {}).get("condition") or []
    if not condition_paths:
        raise RuntimeError(f"{row.get('sample_id')}: no condition frames")
    image = read_rgb(cv2, Path(condition_paths[-1]))
    if image is None:
        raise RuntimeError(f"{row.get('sample_id')}: unreadable last condition")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cannot write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mode", choices=["seedance2_fast_ref", "seedance2_ref", "seedance1_pro_i2v"], default="seedance2_fast_ref")
    parser.add_argument("--resolution", choices=["480p", "720p", "1080p"], default="480p")
    parser.add_argument("--duration", default="4")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set. Export your fal.ai API key before running this gate.")
    try:
        import fal_client
    except ImportError as exc:
        raise RuntimeError("Missing fal-client. Install with: pip install fal-client") from exc

    cv2, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = args.out_dir / "conditioning"
    raw_dir = args.out_dir / "raw"

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"]
    rows = rows[args.offset : args.offset + args.limit]
    if not rows:
        raise RuntimeError("no BDD rows selected")

    report_rows = []
    selected_rows = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        print(f"[{position}/{len(rows)}] {sample_id} | submit {args.mode}", flush=True)
        try:
            if args.mode in {"seedance2_fast_ref", "seedance2_ref"}:
                history_path = media_dir / f"{sample_id}_history.mp4"
                write_history_clip(cv2, row, history_path)
                result = run_fal_seedance2_ref(
                    fal_client,
                    history_path,
                    args.prompt,
                    args.resolution,
                    args.duration,
                    args.seed + position - 1,
                    fast=args.mode == "seedance2_fast_ref",
                )
            else:
                image_path = media_dir / f"{sample_id}_last.png"
                write_last_condition_image(cv2, row, image_path)
                result = run_fal_seedance1_i2v(
                    fal_client,
                    image_path,
                    args.prompt,
                    args.resolution,
                    args.duration,
                    args.seed + position - 1,
                )
            video_url = result["video"]["url"]
            out_video = raw_dir / f"{sample_id}.mp4"
            download(video_url, out_video)
            report_rows.append(
                {
                    "sample_id": sample_id,
                    "video": str(out_video),
                    "status": "ok",
                    "mode": args.mode,
                    "resolution": args.resolution,
                    "duration": args.duration,
                    "seed": result.get("seed", args.seed + position - 1),
                    "url": video_url,
                }
            )
            selected_rows.append(row)
            print(f"[{position}/{len(rows)}] {sample_id} | ok {out_video}", flush=True)
        except Exception as exc:
            report_rows.append(
                {
                    "sample_id": sample_id,
                    "video": "",
                    "status": "bad",
                    "mode": args.mode,
                    "resolution": args.resolution,
                    "duration": args.duration,
                    "seed": args.seed + position - 1,
                    "url": "",
                    "error": str(exc),
                }
            )
            print(f"[{position}/{len(rows)}] {sample_id} | bad {exc}", flush=True)
        time.sleep(1)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "video", "status", "mode", "resolution", "duration", "seed", "url", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for row in report_rows if row["status"] == "ok")
    print(json.dumps({"rows": len(report_rows), "ok": ok, "out_dir": str(args.out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
