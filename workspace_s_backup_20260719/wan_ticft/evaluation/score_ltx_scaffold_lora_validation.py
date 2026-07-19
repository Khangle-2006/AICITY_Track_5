"""Score LTX scaffold-LoRA validation samples against target videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from baseline.frame_io import import_cv2_np
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.audit_perceptual_video_distribution import frechet_proxy


def read_report(path: Path) -> dict[str, Path]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: Path(row["video"])
            for row in csv.DictReader(handle)
            if row.get("sample_id") and row.get("video") and row.get("status", "ok") == "ok"
        }


def read_video(cv2, path: Path, limit: int):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def resize_like(cv2, frame, target):
    height, width = target.shape[:2]
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_json", type=Path, required=True)
    parser.add_argument("--target_report", type=Path, required=True)
    parser.add_argument("--scaffold_report", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 100, 200])
    args = parser.parse_args()
    args.steps = list(dict.fromkeys(args.steps))

    cv2, np = import_cv2_np()
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18

    validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
    targets = read_report(args.target_report)
    scaffolds = read_report(args.scaffold_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidate_names = ["scaffold", *[f"step{step}" for step in args.steps]]
    pixel_rows = []
    all_frames = defaultdict(list)
    all_clips = defaultdict(list)

    for index, item in enumerate(validation):
        sample_id = item["sample_id"]
        paths = {
            "target": targets.get(sample_id),
            "scaffold": scaffolds.get(sample_id),
            **{
                f"step{step}": args.samples / f"step_{step:06d}_{index}.mp4"
                for step in args.steps
            },
        }
        missing = [name for name, path in paths.items() if path is None or not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{sample_id}: missing {missing}")

        videos = {name: read_video(cv2, Path(path), args.frames) for name, path in paths.items()}
        short = {name: len(frames) for name, frames in videos.items() if len(frames) < args.frames}
        if short:
            raise RuntimeError(f"{sample_id}: expected {args.frames} frames, got {short}")

        target = videos["target"]
        all_frames["target"].extend(target)
        all_clips["target"].append(target)
        for name in candidate_names:
            aligned = [resize_like(cv2, frame, gt) for frame, gt in zip(videos[name], target)]
            pixel_rows.append(
                {
                    "sample_id": sample_id,
                    "candidate": name,
                    "frames": args.frames,
                    "psnr": f"{np.mean([psnr(cv2, np, p, t) for p, t in zip(aligned, target)]):.6f}",
                    "ssim": f"{np.mean([ssim_gray(cv2, np, p, t) for p, t in zip(aligned, target)]):.6f}",
                    "absdiff": f"{np.mean([mean_absdiff(np, p, t) for p, t in zip(aligned, target)]):.6f}",
                }
            )
            all_frames[name].extend(aligned)
            all_clips[name].append(aligned)

        positions = sorted(set([0, args.frames // 2, args.frames - 1]))
        sheet_rows = []
        for name in ("target", *candidate_names):
            tiles = []
            for frame_index in positions:
                frame = resize_like(cv2, videos[name][frame_index], target[frame_index])
                tile = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
                label = f"{name} f{frame_index}"
                cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                tiles.append(tile)
            sheet_rows.append(np.concatenate(tiles, axis=1))
        sheet = np.concatenate(sheet_rows, axis=0)
        cv2.imwrite(str(args.out_dir / f"{sample_id}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    write_csv(args.out_dir / "pixel.csv", pixel_rows)
    print(f"{'Candidate':<20} {'N':>3} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 58)
    for name in candidate_names:
        rows = [row for row in pixel_rows if row["candidate"] == name]
        print(
            f"{name:<20} {len(rows):>3} "
            f"{np.mean([float(row['psnr']) for row in rows]):>10.4f} "
            f"{np.mean([float(row['ssim']) for row in rows]):>10.4f} "
            f"{np.mean([float(row['absdiff']) for row in rows]):>10.4f}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    image_model.fc = torch.nn.Identity()
    image_model.eval().to(device)
    video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    video_model.fc = torch.nn.Identity()
    video_model.eval().to(device)

    def image_features(frames):
        output = []
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
        for frame in frames:
            tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            with torch.no_grad():
                output.append(image_model((tensor - mean) / std).cpu())
        return torch.cat(output)

    def video_features(clips):
        output = []
        mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
        std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]
        for frames in clips:
            tensor = torch.stack([torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255) for frame in frames])
            tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
            tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            with torch.no_grad():
                output.append(video_model((tensor - mean) / std).cpu())
        return torch.cat(output)

    target_image_features = image_features(all_frames["target"])
    target_video_features = video_features(all_clips["target"])
    proxy_rows = []
    for name in candidate_names:
        proxy_rows.append(
            {
                "candidate": name,
                "frames": len(all_frames[name]),
                "clips": len(all_clips[name]),
                "fid_proxy": f"{frechet_proxy(torch, target_image_features, image_features(all_frames[name])):.6f}",
                "fvd_proxy": f"{frechet_proxy(torch, target_video_features, video_features(all_clips[name])):.6f}",
            }
        )
    write_csv(args.out_dir / "proxy.csv", proxy_rows)
    print(f"\n{'Candidate':<20} {'FID proxy':>12} {'FVD proxy':>12}")
    print("-" * 46)
    for row in sorted(proxy_rows, key=lambda value: float(value["fid_proxy"]) + float(value["fvd_proxy"])):
        print(f"{row['candidate']:<20} {float(row['fid_proxy']):>12.4f} {float(row['fvd_proxy']):>12.4f}")
    print(f"\noutput={args.out_dir}")


if __name__ == "__main__":
    main()
