"""Gate chunk-detail donor variants on BDD24.

The idea is to preserve the base video's low-frequency geometry while borrowing
only high-frequency detail from a sharper chunked LTX output.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_manifest(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def read_video(path: Path, count: int, size: tuple[int, int] | None = None, pad_last: bool = True) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if size is not None and (rgb.shape[1], rgb.shape[0]) != size:
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        frames.append(rgb)
    cap.release()
    if pad_last and frames:
        while len(frames) < count:
            frames.append(frames[-1].copy())
    return frames


def read_targets(item: dict) -> list[np.ndarray]:
    return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in item["frames"]["target"]]


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)
    return float(99.0 if mse <= 1e-8 else 20.0 * np.log10(255.0 / np.sqrt(mse)))


def mean_absdiff(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))


def ssim_gray(pred: np.ndarray, target: np.ndarray) -> float:
    x = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).astype(np.float32)
    y = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return float(np.mean(score))


def blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigma)


def donor_mask(base: np.ndarray, donor: np.ndarray, sigma: float, threshold: float) -> np.ndarray:
    base_low = blur(base, sigma)
    donor_low = blur(donor, sigma)
    diff = np.mean(np.abs(base_low - donor_low), axis=2)
    mask = np.clip((threshold - diff) / max(threshold, 1e-6), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
    return mask[..., None].astype(np.float32)


def detail_donor(base: np.ndarray, donor: np.ndarray, alpha: float, sigma: float, threshold: float) -> np.ndarray:
    b = base.astype(np.float32)
    d = donor.astype(np.float32)
    detail = d - blur(d, sigma)
    mask = donor_mask(b, d, sigma=sigma * 1.4, threshold=threshold)
    return np.clip(b + alpha * detail * mask, 0, 255).astype(np.uint8)


def lowfreq_guarded_mix(base: np.ndarray, donor: np.ndarray, alpha: float, sigma: float, threshold: float) -> np.ndarray:
    b = base.astype(np.float32)
    d = donor.astype(np.float32)
    b_low = blur(b, sigma)
    d_low = blur(d, sigma)
    d_matched = d + (b_low - d_low)
    mask = donor_mask(b, d, sigma=sigma, threshold=threshold)
    return np.clip(b * (1 - alpha * mask) + d_matched * (alpha * mask), 0, 255).astype(np.uint8)


def make_sheet(path: Path, rows: list[tuple[str, list[np.ndarray]]], positions: list[int]) -> None:
    tiles_rows = []
    for label, frames in rows:
        tiles = []
        for pos in positions:
            frame = frames[min(pos, len(frames) - 1)]
            tile = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"{label} f{pos}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(tile, f"{label} f{pos}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        tiles_rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(tiles_rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def image_video_proxy(all_frames: dict[str, list[np.ndarray]], all_clips: dict[str, list[list[np.ndarray]]], out_csv: Path) -> None:
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18
    from wan_ticft.evaluation.audit_perceptual_video_distribution import frechet_proxy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    image_model.fc = torch.nn.Identity()
    image_model.eval().to(device)
    video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    video_model.fc = torch.nn.Identity()
    video_model.eval().to(device)

    def image_features(frames: list[np.ndarray]) -> torch.Tensor:
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
        feats = []
        for frame in frames:
            tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            with torch.no_grad():
                feats.append(image_model((tensor - mean) / std).cpu())
        return torch.cat(feats)

    def video_features(clips: list[list[np.ndarray]]) -> torch.Tensor:
        mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
        std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]
        feats = []
        for frames in clips:
            tensor = torch.stack([torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255) for frame in frames])
            tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
            tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            with torch.no_grad():
                feats.append(video_model((tensor - mean) / std).cpu())
        return torch.cat(feats)

    target_img = image_features(all_frames["target"])
    target_vid = video_features(all_clips["target"])
    rows = []
    for name in sorted(k for k in all_frames if k != "target"):
        rows.append({
            "candidate": name,
            "frames": len(all_frames[name]),
            "clips": len(all_clips[name]),
            "fid_proxy": f"{frechet_proxy(torch, target_img, image_features(all_frames[name])):.6f}",
            "fvd_proxy": f"{frechet_proxy(torch, target_vid, video_features(all_clips[name])):.6f}",
        })
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--donor_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--base_ext", default=".mp4")
    parser.add_argument("--donor_ext", default=".mp4")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    items = read_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "base": lambda b, d: b,
        "donor": lambda b, d: d,
        "detail_a035": lambda b, d: detail_donor(b, d, 0.35, 1.15, 18.0),
        "detail_a055": lambda b, d: detail_donor(b, d, 0.55, 1.15, 18.0),
        "matched_mix_a020": lambda b, d: lowfreq_guarded_mix(b, d, 0.20, 3.0, 14.0),
        "matched_mix_a035": lambda b, d: lowfreq_guarded_mix(b, d, 0.35, 3.0, 14.0),
    }
    all_frames = defaultdict(list)
    all_clips = defaultdict(list)
    pixel_rows = []

    for item in items:
        sample_id = item["sample_id"]
        targets = read_targets(item)
        size = (targets[0].shape[1], targets[0].shape[0])
        base_path = args.base_dir / f"{sample_id}{args.base_ext}"
        donor_path = args.donor_dir / f"{sample_id}{args.donor_ext}"
        base = read_video(base_path, len(targets), size)
        donor = read_video(donor_path, len(targets), size)
        if len(base) < len(targets) or len(donor) < len(targets):
            raise RuntimeError(f"{sample_id}: short read base={len(base)} donor={len(donor)} target={len(targets)}")

        fps = float(item["frame_indices"].get("fps", 30.0))
        made = {}
        for name, func in variants.items():
            frames = [func(b, d) for b, d in zip(base, donor)]
            made[name] = frames
            write_video(args.out_dir / name / f"{sample_id}.mp4", frames, fps)
            all_frames[name].extend(frames)
            all_clips[name].append(frames)
            pixel_rows.append({
                "sample_id": sample_id,
                "candidate": name,
                "psnr": f"{np.mean([psnr(p, t) for p, t in zip(frames, targets)]):.6f}",
                "ssim": f"{np.mean([ssim_gray(p, t) for p, t in zip(frames, targets)]):.6f}",
                "absdiff": f"{np.mean([mean_absdiff(p, t) for p, t in zip(frames, targets)]):.6f}",
            })
        all_frames["target"].extend(targets)
        all_clips["target"].append(targets)
        if len(list((args.out_dir / "sheets").glob("*.jpg"))) < 8:
            make_sheet(
                args.out_dir / "sheets" / f"{sample_id}.jpg",
                [("target", targets), ("base", made["base"]), ("donor", made["donor"]), ("detail55", made["detail_a055"]), ("mix35", made["matched_mix_a035"])],
                [0, len(targets) // 2, len(targets) - 1],
            )

    with (args.out_dir / "pixel.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pixel_rows[0]))
        writer.writeheader()
        writer.writerows(pixel_rows)
    image_video_proxy(all_frames, all_clips, args.out_dir / "proxy.csv")
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()
