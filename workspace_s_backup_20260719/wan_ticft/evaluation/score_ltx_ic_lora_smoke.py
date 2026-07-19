"""Score the existing IC-LoRA validation samples against their paired targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from baseline.frame_io import import_cv2_np
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.audit_perceptual_video_distribution import frechet_proxy


def read_video(cv2, path: Path, limit: int = 9):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < limit:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
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
    parser.add_argument("--region_base", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18

    validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = ("region_medium", "step0", "step100", "step200")
    pixel_rows = []
    all_frames = defaultdict(list)
    all_clips = defaultdict(list)

    for index, item in enumerate(validation):
        sample_id = item["sample_id"]
        paths = {
            "target": args.region_base / "target" / f"{sample_id}.mp4",
            "region_medium": args.region_base / "region_norm_medium" / f"{sample_id}.mp4",
            "step0": args.samples / f"step_000000_{index}.mp4",
            "step100": args.samples / f"step_000100_{index}.mp4",
            "step200": args.samples / f"step_000200_{index}.mp4",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("\n".join(missing))
        videos = {name: read_video(cv2, path) for name, path in paths.items()}
        if any(len(frames) != 9 for frames in videos.values()):
            counts = {name: len(frames) for name, frames in videos.items()}
            raise RuntimeError(f"{sample_id}: expected 9 readable frames, got {counts}")

        target = videos["target"]
        all_frames["target"].extend(target)
        all_clips["target"].append(target)
        for name in candidates:
            aligned = [resize_like(cv2, frame, gt) for frame, gt in zip(videos[name], target)]
            pixel_rows.append(
                {
                    "sample_id": sample_id,
                    "candidate": name,
                    "frames": 9,
                    "psnr": f"{np.mean([psnr(cv2, np, p, t) for p, t in zip(aligned, target)]):.6f}",
                    "ssim": f"{np.mean([ssim_gray(cv2, np, p, t) for p, t in zip(aligned, target)]):.6f}",
                    "absdiff": f"{np.mean([mean_absdiff(np, p, t) for p, t in zip(aligned, target)]):.6f}",
                }
            )
            all_frames[name].extend(aligned)
            all_clips[name].append(aligned)

        positions = (0, 4, 8)
        sheet_rows = []
        for name in ("target", *candidates):
            tiles = []
            for frame_index in positions:
                frame = resize_like(cv2, videos[name][frame_index], target[frame_index])
                tile = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
                cv2.putText(tile, f"{name} f{frame_index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(tile, f"{name} f{frame_index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                tiles.append(tile)
            sheet_rows.append(np.concatenate(tiles, axis=1))
        sheet = np.concatenate(sheet_rows, axis=0)
        cv2.imwrite(str(args.out_dir / f"{sample_id}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    write_csv(args.out_dir / "pixel.csv", pixel_rows)
    print(f"{'Candidate':<20} {'N':>3} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 58)
    for name in candidates:
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
    for name in candidates:
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
