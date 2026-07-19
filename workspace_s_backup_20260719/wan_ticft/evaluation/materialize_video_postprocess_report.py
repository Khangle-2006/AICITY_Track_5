from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_report(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("sample_id") and row.get("video") and Path(row["video"]).exists()
        ]


def unsharp(cv2, np, bgr, sigma: float, amount: float):
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip((1.0 + amount) * bgr.astype(np.float32) - amount * blur.astype(np.float32), 0, 255).astype(
        np.uint8
    )


def clahe_luma(cv2, bgr, clip_limit: float, tile_grid: int):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    luma = clahe.apply(luma)
    return cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2BGR)


def apply_postprocess(cv2, np, bgr, mode: str, sigma: float, amount: float, clahe_clip: float, tile_grid: int):
    if mode == "unsharp":
        return unsharp(cv2, np, bgr, sigma, amount)
    if mode == "clahe":
        return clahe_luma(cv2, bgr, clahe_clip, tile_grid)
    if mode == "unsharp_clahe":
        return clahe_luma(cv2, unsharp(cv2, np, bgr, sigma, amount), clahe_clip, tile_grid)
    raise ValueError(f"unknown mode: {mode}")


def process_video(cv2, np, source: Path, destination: Path, mode: str, sigma: float, amount: float, clahe_clip: float, tile_grid: int):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"bad video shape: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot create {destination}")
    frames = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        writer.write(apply_postprocess(cv2, np, bgr, mode, sigma, amount, clahe_clip, tile_grid))
        frames += 1
    cap.release()
    writer.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize postprocessed videos from a sample_id/video report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=["unsharp", "clahe", "unsharp_clahe"], required=True)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--amount", type=float, default=0.25)
    parser.add_argument("--clahe_clip", type=float, default=1.2)
    parser.add_argument("--tile_grid", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    from baseline.frame_io import import_cv2_np

    cv2, np = import_cv2_np()
    rows = read_report(args.report)
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    output = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        source = Path(row["video"])
        destination = videos_dir / f"{sample_id}.mp4"
        frames, fps = process_video(
            cv2, np, source, destination, args.mode, args.sigma, args.amount, args.clahe_clip, args.tile_grid
        )
        output.append(
            {
                "sample_id": sample_id,
                "video": str(destination),
                "status": "ok",
                "source_video": str(source),
                "source_fps": f"{fps:.6f}",
                "future_frames": frames,
                "label": args.label,
                "mode": args.mode,
                "sigma": args.sigma,
                "amount": args.amount,
                "clahe_clip": args.clahe_clip,
                "tile_grid": args.tile_grid,
            }
        )
        print(f"[{position}/{len(rows)}] {sample_id} | ok", flush=True)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
