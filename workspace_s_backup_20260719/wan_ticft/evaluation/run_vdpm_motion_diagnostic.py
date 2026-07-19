from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path


def extract_frames(cv2, video_path: Path, out_dir: Path, max_frames: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        path = out_dir / f"{len(frames):06d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"cannot write {path}")
        frames.append(path)
    cap.release()
    return frames


def save_visuals(cv2, np, score, confidence, image_rgb, out_dir: Path, stem: str = "motion") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_vis = np.clip(score / max(float(np.percentile(score, 95)), 1e-6), 0.0, 1.0)
    heat = cv2.applyColorMap((score_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image_bgr = cv2.cvtColor((np.clip(image_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if heat.shape[:2] != image_bgr.shape[:2]:
        heat = cv2.resize(heat, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.addWeighted(image_bgr, 0.55, heat, 0.45, 0)
    conf_vis = np.clip(
        confidence / max(float(np.percentile(confidence, 95)), 1e-6), 0.0, 1.0
    )
    cv2.imwrite(str(out_dir / f"{stem}_heatmap.png"), heat)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
    if stem == "motion":
        cv2.imwrite(str(out_dir / "confidence.png"), (conf_vis * 255).astype(np.uint8))


def fit_rigid(torch, source, target, weights):
    weights = weights / weights.sum().clamp_min(1e-8)
    source_mean = (source * weights[:, None]).sum(dim=0)
    target_mean = (target * weights[:, None]).sum(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (source_centered * weights[:, None]).T @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    translation = target_mean - source_mean @ rotation
    return rotation, translation


def rigid_residual_motion(torch, tracks, confidence, keep_fraction: float = 0.70):
    """Remove the dominant camera-induced rigid transform from each pointmap."""
    reference_index = tracks.shape[0] - 1
    reference = tracks[reference_index].reshape(-1, 3).float()
    weights = confidence.reshape(-1).float()
    valid_base = torch.isfinite(reference).all(dim=1) & torch.isfinite(weights)
    valid_base &= weights >= torch.quantile(weights[torch.isfinite(weights)], 0.25)
    residual_maps = []
    for index, value in enumerate(tracks):
        if index == reference_index:
            residual_maps.append(torch.zeros(tracks.shape[1:3], dtype=torch.float32))
            continue
        source = value.reshape(-1, 3).float()
        valid = valid_base & torch.isfinite(source).all(dim=1)
        inliers = valid.clone()
        residual = torch.zeros(source.shape[0], dtype=torch.float32)
        for _ in range(3):
            selected = torch.where(inliers)[0]
            if selected.numel() < 32:
                break
            # A bounded deterministic subset keeps the CPU SVD fit inexpensive.
            if selected.numel() > 50000:
                selected = selected[:: max(1, selected.numel() // 50000)][:50000]
            rotation, translation = fit_rigid(
                torch,
                source[selected],
                reference[selected],
                weights[selected],
            )
            aligned = source @ rotation + translation
            residual = torch.linalg.vector_norm(aligned - reference, dim=1)
            threshold = torch.quantile(residual[valid], keep_fraction)
            inliers = valid & (residual <= threshold)
        residual[~valid] = 0
        residual_maps.append(residual.view(tracks.shape[1:3]))
    return torch.stack(residual_maps).amax(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vdpm_repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()

    sys.path.insert(0, str(args.vdpm_repo))
    import cv2
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from vggt.utils.load_fn import load_and_preprocess_images

    from dpm.model import VDPM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with initialize_config_dir(config_dir=str(args.vdpm_repo / "configs"), version_base=None):
        cfg = compose(config_name="visualise")
    model = VDPM(cfg).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(model.load_state_dict(state, strict=True), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for position, video_path in enumerate(sorted(args.input_dir.glob("*.mp4")), start=1):
        sample_id = video_path.stem
        sample_dir = args.out_dir / sample_id
        frame_paths = extract_frames(cv2, video_path, sample_dir / "input_frames", args.max_frames)
        if len(frame_paths) < 2:
            raise RuntimeError(f"{sample_id}: fewer than two frames")
        images = load_and_preprocess_images([str(path) for path in frame_paths]).cuda()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.precision == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            predictions = model.inference(None, images=images.unsqueeze(0))

        reference = len(frame_paths) - 1
        points = torch.cat([item["pts3d"].float().cpu() for item in predictions["pointmaps"]], dim=0)
        confidence = torch.cat([item["conf"].float().cpu() for item in predictions["pointmaps"]], dim=0)
        tracks = points[:, reference]
        raw_score = torch.linalg.vector_norm(tracks - tracks[reference : reference + 1], dim=-1).amax(dim=0)
        conf = confidence[:, reference].mean(dim=0)
        score = rigid_residual_motion(torch, tracks, conf)
        valid = conf >= torch.quantile(conf.flatten(), 0.25)
        valid_score = score[valid]
        valid_raw_score = raw_score[valid]

        score_np = score.numpy()
        raw_score_np = raw_score.numpy()
        conf_np = conf.numpy()
        image_rgb = images[reference].permute(1, 2, 0).cpu().numpy()
        save_visuals(cv2, np, raw_score_np, conf_np, image_rgb, sample_dir, stem="raw_motion")
        save_visuals(cv2, np, score_np, conf_np, image_rgb, sample_dir, stem="motion")
        raw_p90 = float(torch.quantile(valid_raw_score, 0.90))
        residual_p90 = float(torch.quantile(valid_score, 0.90))
        row = {
            "sample_id": sample_id,
            "frames": len(frame_paths),
            "height": int(score.shape[0]),
            "width": int(score.shape[1]),
            "motion_median": float(torch.median(valid_score)),
            "motion_p90": float(torch.quantile(valid_score, 0.90)),
            "motion_p99": float(torch.quantile(valid_score, 0.99)),
            "raw_motion_p90": raw_p90,
            "residual_ratio_p90": residual_p90 / max(raw_p90, 1e-8),
            "dynamic_fraction_p90": float(((score >= torch.quantile(valid_score, 0.90)) & valid).float().mean()),
            "valid_fraction": float(valid.float().mean()),
            "overlay": str(sample_dir / "motion_overlay.png"),
        }
        report.append(row)
        print(f"[{position}] {sample_id} | p90={row['motion_p90']:.6f}", flush=True)
        del predictions, points, confidence, tracks, images
        torch.cuda.empty_cache()

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
