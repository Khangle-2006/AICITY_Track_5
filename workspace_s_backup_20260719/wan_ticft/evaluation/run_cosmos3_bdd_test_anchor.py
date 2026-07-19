from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb
from baseline.verify import verify_sample_output


FIELDS = [
    "sample_id",
    "output_name",
    "video",
    "status",
    "job_id",
    "target_N",
    "future_frames",
    "source_fps",
    "peak_memory_mb",
    "inference_time_s",
    "notes",
]


def read_rows(path: Path, limit: int, offset: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("view_type") == "bdd"]
    return rows[offset : offset + limit if limit else None]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    tmp.replace(path)


def prompt(row: dict[str, str]) -> str:
    return " ".join(
        [
            "Use the reference image as a strict first-frame anchor.",
            "Continue the exact same forward-facing driving camera scene.",
            "Preserve the same road, buildings, vehicles, pedestrians, windshield, dashboard, camera angle, horizon, lighting, and background layout.",
            "Only advance existing visible objects slightly in time.",
            "Do not add new people or vehicles. Do not change the location.",
            "Keep background stable, realistic, sharp, and temporally consistent.",
        ]
    )


def run_json(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed {proc.returncode}: {' '.join(command)}\n{proc.stderr}\n{proc.stdout}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"expected JSON:\n{proc.stdout}\n{proc.stderr}") from exc


def submit(args, image_path: Path, row: dict[str, str], frames: int, seed: int) -> str:
    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": False,
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
        f"prompt={prompt(row)}",
        "-F",
        "negative_prompt=new scene, different street, changed buildings, changed cars, changed pedestrians, tilted camera, rotated horizon, blur, ghosting, double exposure, warped geometry, hallucinated objects, low quality, jittery, deformed",
        "-F",
        f"size={args.width}x{args.height}",
        "-F",
        f"num_frames={frames}",
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
    response = run_json(command)
    if response.get("error"):
        raise RuntimeError(json.dumps(response, indent=2))
    job_id = str(response.get("id") or "")
    if not job_id:
        raise RuntimeError(f"missing job id: {json.dumps(response, indent=2)}")
    return job_id


def poll(args, job_id: str) -> dict[str, Any]:
    deadline = time.time() + args.timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = run_json(["curl", "-sS", f"{args.server}/v1/videos/{job_id}"])
        status = last.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return last
        print(f"{job_id}: status={status} progress={last.get('progress')}", flush=True)
        time.sleep(args.poll_interval)
    raise TimeoutError(f"{job_id}: timed out; last={json.dumps(last, indent=2)}")


def download(args, job_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sS", "-fL", f"{args.server}/v1/videos/{job_id}/content", "-o", str(out_path)],
        check=True,
    )
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"{job_id}: bad download {out_path}")


def write_anchor(cv2, row: dict[str, str], path: Path) -> None:
    history_paths = numeric_pngs(Path(row["input_dir"]))
    if not history_paths:
        raise RuntimeError(f"{row['sample_id']}: no input history frames")
    image = read_rgb(cv2, history_paths[-1])
    if image is None:
        raise RuntimeError(f"{row['sample_id']}: cannot read {history_paths[-1]}")
    if image.shape[1] != 1280 or image.shape[0] != 720:
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cannot write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cosmos3 image-anchor videos for BDD test rows.")
    parser.add_argument("--test_csv", type=Path, default=Path("/workspace/experiments/index/test_index.csv"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--flow_shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max_frames", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--poll_interval", type=float, default=5.0)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = read_rows(args.test_csv, args.limit, args.offset)
    if not rows:
        raise RuntimeError("no BDD test rows selected")
    if args.clean and args.out_dir.exists() and not args.resume:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.out_dir / "report.csv"
    report: list[dict[str, Any]] = []
    if args.resume and report_path.exists():
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            report = list(csv.DictReader(handle))
    done = {row["sample_id"] for row in report if row.get("status") == "ok" and Path(row.get("video", "")).exists()}

    for pos, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        frames = min(args.max_frames, target_n)
        out_video = args.out_dir / "videos" / f"{output_name}.mp4"
        if args.resume and sample_id in done and out_video.exists():
            print(f"[{pos}/{len(rows)}] {output_name} | resume", flush=True)
            continue

        anchor = args.out_dir / "anchor_images" / f"{output_name}.png"
        write_anchor(cv2, row, anchor)
        print(f"[{pos}/{len(rows)}] {output_name} | frames={frames} target={target_n}", flush=True)
        job_id = submit(args, anchor, row, frames, args.seed + args.offset + pos - 1)
        status = poll(args, job_id)
        if status.get("status") != "completed":
            raise RuntimeError(f"{sample_id}: job failed: {json.dumps(status, indent=2)}")
        download(args, job_id, out_video)
        report.append(
            {
                "sample_id": sample_id,
                "output_name": output_name,
                "video": str(out_video),
                "status": "ok",
                "job_id": job_id,
                "target_N": target_n,
                "future_frames": frames,
                "source_fps": args.fps,
                "peak_memory_mb": status.get("peak_memory_mb", ""),
                "inference_time_s": status.get("inference_time_s", ""),
                "notes": "",
            }
        )
        write_csv(report_path, report)

    write_csv(report_path, report)
    print(json.dumps({"rows": len(report), "status": "ok", "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
