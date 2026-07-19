"""Compare validation candidates with feature-distribution proxies for FID/FVD."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def load_targets(cv2, row):
    paths = (row.get("frames") or {}).get("target") or []
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def load_candidate(cv2, row, report: dict[str, Path]):
    video = report.get(row["sample_id"])
    if video is None:
        return []
    mapped = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, mapped)
    return [frame_map[index] for index in mapped if index in frame_map]


def load_full_video(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def covariance(torch, values):
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(1, values.shape[0] - 1)


def frechet_proxy(torch, first, second):
    if first.shape[1] > 256:
        generator = torch.Generator().manual_seed(17)
        projection = torch.randn(first.shape[1], 256, generator=generator) / first.shape[1] ** 0.5
        first = first @ projection
        second = second @ projection
    first = first.double()
    second = second.double()
    mean_delta = (first.mean(dim=0) - second.mean(dim=0)).pow(2).sum()
    cov_first = covariance(torch, first)
    cov_second = covariance(torch, second)
    eigenvalues = torch.linalg.eigvals(cov_first @ cov_second).real.clamp_min(0)
    trace_sqrt = eigenvalues.sqrt().sum()
    value = mean_delta + torch.trace(cov_first) + torch.trace(cov_second) - 2.0 * trace_sqrt
    return float(value.clamp_min(0).cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, help="NAME=CSV")
    parser.add_argument("--target_report", type=Path, default=None)
    parser.add_argument("--full_video", action="store_true")
    parser.add_argument("--view_type", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--clip_frames", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=4)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_weights = Inception_V3_Weights.DEFAULT
    video_weights = R3D_18_Weights.DEFAULT
    image_model = inception_v3(weights=image_weights)
    image_model.fc = torch.nn.Identity()
    image_model.eval().to(device)
    video_model = r3d_18(weights=video_weights)
    video_model.fc = torch.nn.Identity()
    video_model.eval().to(device)

    reports = {}
    for item in args.report:
        name, path = item.split("=", 1)
        reports[name] = read_report(Path(path))
    target_report = read_report(args.target_report) if args.target_report else None

    frame_features = defaultdict(list)
    video_features = defaultdict(list)

    def image_feature(frame):
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
        with torch.no_grad():
            return image_model((tensor - mean) / std).cpu()

    def video_feature(frames):
        indices = np.linspace(0, len(frames) - 1, args.clip_frames).round().astype(int)
        tensor = torch.stack(
            [torch.from_numpy(frames[index].copy()).permute(2, 0, 1).float().div(255) for index in indices]
        )
        tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
        std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]
        with torch.no_grad():
            return video_model((tensor - mean) / std).cpu()

    rows = read_jsonl(args.manifest)
    if args.view_type:
        rows = [row for row in rows if row.get("view_type") == args.view_type]
    if args.phase:
        rows = [row for row in rows if str(row.get("phase", "")) == str(args.phase)]
    for position, row in enumerate(rows, start=1):
        if args.full_video:
            target_path = target_report.get(row["sample_id"]) if target_report else None
            candidates = {"target": load_full_video(cv2, target_path) if target_path else []}
            candidates.update(
                {
                    name: load_full_video(cv2, report[row["sample_id"]])
                    if row["sample_id"] in report
                    else []
                    for name, report in reports.items()
                }
            )
        else:
            candidates = {"target": load_targets(cv2, row)}
            candidates.update({name: load_candidate(cv2, row, report) for name, report in reports.items()})
        target_count = len(candidates["target"])
        if target_count < args.clip_frames or any(len(frames) != target_count for frames in candidates.values()):
            print(f"[{position}] {row['sample_id']} | skipped", flush=True)
            continue
        for name, frames in candidates.items():
            for frame in frames[:: args.frame_stride]:
                frame_features[name].append(image_feature(frame))
            video_features[name].append(video_feature(frames))
        print(f"[{position}] {row['sample_id']} | ok", flush=True)

    missing = [name for name in ("target", *reports) if not frame_features[name] or not video_features[name]]
    if missing:
        raise RuntimeError(f"no complete feature samples for: {', '.join(missing)}")

    target_frames = torch.cat(frame_features["target"])
    target_videos = torch.cat(video_features["target"])
    output = []
    for name in reports:
        output.append(
            {
                "candidate": name,
                "frames": len(frame_features[name]),
                "clips": len(video_features[name]),
                "fid_proxy": f"{frechet_proxy(torch, target_frames, torch.cat(frame_features[name])):.6f}",
                "fvd_proxy": f"{frechet_proxy(torch, target_videos, torch.cat(video_features[name])):.6f}",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"{'Candidate':<24} {'Frames':>8} {'Clips':>7} {'FID proxy':>12} {'FVD proxy':>12}")
    print("-" * 70)
    for row in sorted(output, key=lambda item: float(item["fid_proxy"]) + float(item["fvd_proxy"])):
        print(
            f"{row['candidate']:<24} {int(row['frames']):>8} {int(row['clips']):>7} "
            f"{float(row['fid_proxy']):>12.4f} {float(row['fvd_proxy']):>12.4f}"
        )
    print(f"output={args.out_csv}")


if __name__ == "__main__":
    main()
