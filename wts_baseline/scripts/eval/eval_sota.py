"""
SOTA Enhanced Evaluation Script for AI City Challenge 2026 Track 5

Computes all 5 competition metrics:
1. PSNR - Peak Signal-to-Noise Ratio (pixel-level fidelity)
2. SSIM - Structural Similarity Index Measure (structural fidelity)
3. LPIPS - Learned Perceptual Image Patch Similarity (perceptual quality)
4. CLIP-S - CLIP Score (text-video alignment)
5. FVD - Frechet Video Distance (temporal coherence)

Also computes the unweighted average across all metrics.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    return psnr_fn(img1, img2, data_range=255)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images."""
    return ssim_fn(img1, img2, data_range=255, channel_axis=-1, win_size=7)


def compute_lpips(img1: torch.Tensor, img2: torch.Tensor, model) -> float:
    """Compute LPIPS between two image tensors."""
    img1 = img1.to(next(model.parameters()).device)
    img2 = img2.to(next(model.parameters()).device)
    return model(img1, img2).item()


def compute_clip_score(
    images: list,
    caption: str,
    clip_model,
    clip_processor,
    device: torch.device,
) -> float:
    """Compute CLIP-S between images and caption."""
    inputs = clip_processor(text=caption, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    text_features = clip_model.get_text_features(**inputs)
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    text_features = F.normalize(text_features, dim=-1)

    image_scores = []
    for img in images:
        img_inputs = clip_processor(images=img, return_tensors="pt").to(device)
        img_features = clip_model.get_image_features(**img_inputs)
        if hasattr(img_features, "pooler_output"):
            img_features = img_features.pooler_output
        img_features = F.normalize(img_features, dim=-1)
        score = (img_features @ text_features.T).item()
        image_scores.append(score)

    return np.mean(image_scores)


def compute_fvd(
    generated_videos: list,
    real_videos: list,
    feature_extractor,
    device: torch.device,
) -> float:
    """
    Compute Frechet Video Distance.

    Uses I3D features to measure distribution distance between
    generated and real video sets.
    """
    try:
        from cdfvd.fvd import load_i3d_model, preprocess_i3d
        from cdfvd.third_party.i3d.utils import frechet_distance

        # Determine the target frame length (minimum frame length across all videos)
        min_f = min([len(v) for v in generated_videos] + [len(v) for v in real_videos])
        if min_f <= 0:
            return 0.0

        # Crop all videos to min_f frames to ensure they can be stacked
        gen_videos_cropped = [v[:min_f] for v in generated_videos]
        real_videos_cropped = [v[:min_f] for v in real_videos]

        gen_videos_np = np.stack([np.stack(v) for v in gen_videos_cropped])  # (B, T, H, W, C)
        real_videos_np = np.stack([np.stack(v) for v in real_videos_cropped])  # (B, T, H, W, C)

        # Load pre-trained I3D model
        i3d = load_i3d_model(device)

        # Preprocess videos
        gen_tensor = preprocess_i3d(gen_videos_np, target_resolution=(224, 224)).to(device)
        real_tensor = preprocess_i3d(real_videos_np, target_resolution=(224, 224)).to(device)

        # Interpolate temporally to 9 frames if less than 9 frames (downsampling limit of I3D)
        if gen_tensor.shape[2] < 9:
            import torch.nn.functional as F
            gen_tensor = F.interpolate(gen_tensor, size=(9, 224, 224), mode='trilinear', align_corners=False)
            real_tensor = F.interpolate(real_tensor, size=(9, 224, 224), mode='trilinear', align_corners=False)

        with torch.no_grad():
            gen_feats = i3d(gen_tensor).squeeze()
            real_feats = i3d(real_tensor).squeeze()

        # If batch size is 1, squeeze() results in 1D tensor of shape (400,).
        # We need 2D tensor (1, 400) for covariance calculation.
        if gen_feats.dim() == 1:
            gen_feats = gen_feats.unsqueeze(0)
            real_feats = real_feats.unsqueeze(0)

        fvd_score = frechet_distance(gen_feats, real_feats).item()
        return fvd_score
    except Exception as e:
        print(f"Failed to compute FVD using cdfvd (I3D): {e}. Falling back to simplified frame-level ResNet FVD.")
        return compute_fvd_simple(generated_videos, real_videos, feature_extractor, device)


def compute_fvd_simple(
    generated_videos: list,
    real_videos: list,
    feature_extractor,
    device: torch.device,
) -> float:
    """Simplified FVD using frame-level features."""
    gen_features = []
    for video in generated_videos:
        for frame in video:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                feat = feature_extractor(frame_tensor)
            gen_features.append(feat.cpu().numpy())

    real_features = []
    for video in real_videos:
        for frame in video:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                feat = feature_extractor(frame_tensor)
            real_features.append(feat.cpu().numpy())

    gen_features = np.concatenate(gen_features, axis=0)
    real_features = np.concatenate(real_features, axis=0)

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False) + np.eye(gen_features.shape[1]) * 1e-6
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6

    diff = mu_gen - mu_real
    covmean, _ = _sqrtm(sigma_gen @ sigma_real)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return max(0, fvd)


def _sqrtm(A):
    """Matrix square root using scipy."""
    try:
        from scipy.linalg import sqrtm
        return sqrtm(A), None
    except ImportError:
        return np.eye(A.shape[0]), None


def find_sample_pairs(output_dir: str):
    """Find sample_N / sample_N_gt directory pairs."""
    output_path = Path(output_dir)
    pairs = []

    sample_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    for sample_dir in sample_dirs:
        gt_dir = output_path / f"{sample_dir.name}_gt"
        if gt_dir.exists():
            pairs.append((sample_dir, gt_dir))

    if not pairs:
        all_dirs = sorted([d for d in output_path.iterdir() if d.is_dir()])
        for i in range(0, len(all_dirs) - 1, 2):
            if all_dirs[i].name.endswith("_gt"):
                continue
            potential_gt = all_dirs[i + 1] if i + 1 < len(all_dirs) else None
            if potential_gt and "_gt" in potential_gt.name:
                pairs.append((all_dirs[i], potential_gt))
            else:
                pairs.append((all_dirs[i], None))

    return pairs


def load_frames(directory: Path):
    """Load all frames from a directory."""
    frames = []
    for img_path in sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg")):
        img = np.array(Image.open(img_path).convert("RGB"))
        frames.append(img)
    return frames


def evaluate_metrics(
    output_dir: str,
    manifest_path: str = None,
    use_lpips: bool = True,
    use_clip: bool = True,
    use_fvd: bool = True,
    device: str = "cuda",
):
    """
    Evaluate all metrics for generated samples.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    pairs = find_sample_pairs(output_dir)
    print(f"Found {len(pairs)} sample pairs")

    results = defaultdict(list)
    per_sample_results = {}

    lpips_model = None
    if use_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        except ImportError:
            print("LPIPS not available, skipping.")
            use_lpips = False

    clip_model = None
    clip_processor = None
    if use_clip:
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        except ImportError:
            print("CLIP not available, skipping.")
            use_clip = False

    feature_extractor = None
    if use_fvd:
        feature_extractor = _get_feature_extractor(device)

    manifest_entries = []
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    manifest_entries.append(json.loads(line.strip()))

    for pair_idx, (gen_dir, gt_dir) in enumerate(pairs):
        sample_name = gen_dir.name

        gen_frames = load_frames(gen_dir)
        if not gen_frames:
            continue

        if gt_dir and gt_dir.exists():
            gt_frames = load_frames(gt_dir)
        else:
            gt_frames = None

        sample_metrics = {}

        if gt_frames:
            min_len = min(len(gen_frames), len(gt_frames))
            gen_frames = gen_frames[:min_len]
            gt_frames = gt_frames[:min_len]

            psnr_scores = []
            ssim_scores = []
            lpips_scores = []

            for i in range(min_len):
                psnr_scores.append(compute_psnr(gen_frames[i], gt_frames[i]))
                ssim_scores.append(compute_ssim(gen_frames[i], gt_frames[i]))

                if use_lpips and lpips_model is not None:
                    gen_t = torch.from_numpy(gen_frames[i]).permute(2, 0, 1).float() / 127.5 - 1
                    gt_t = torch.from_numpy(gt_frames[i]).permute(2, 0, 1).float() / 127.5 - 1
                    lpips_scores.append(compute_lpips(gen_t.unsqueeze(0), gt_t.unsqueeze(0), lpips_model))

            sample_metrics["psnr"] = np.mean(psnr_scores)
            sample_metrics["ssim"] = np.mean(ssim_scores)
            sample_metrics["lpips"] = np.mean(lpips_scores) if lpips_scores else 0.0

            results["psnr"].append(sample_metrics["psnr"])
            results["ssim"].append(sample_metrics["ssim"])
            results["lpips"].append(sample_metrics["lpips"])

        if use_clip and clip_model is not None:
            # Find the correct manifest entry for this sample
            entry = None
            for e in manifest_entries:
                if e.get("scenario_name") == sample_name:
                    entry = e
                    break
            if entry is None and sample_name.startswith("sample_"):
                try:
                    idx = int(sample_name.split("_")[1])
                    if idx < len(manifest_entries):
                        entry = manifest_entries[idx]
                except Exception:
                    pass
            if entry is None and pair_idx < len(manifest_entries):
                entry = manifest_entries[pair_idx]

            caption = entry.get("merged_caption", "") if entry is not None else ""
            if caption:
                clip_score = compute_clip_score(gen_frames, caption, clip_model, clip_processor, device)
                sample_metrics["clip_s"] = clip_score
                results["clip_s"].append(clip_score)

        per_sample_results[sample_name] = sample_metrics

    if use_fvd and feature_extractor is not None and len(pairs) > 0:
        gen_videos = []
        real_videos = []

        for gen_d, gt_d in pairs:
            if gt_d and gt_d.exists():
                gen_frames = load_frames(gen_d)
                gt_frames = load_frames(gt_d)
                if gen_frames and gt_frames:
                    gen_videos.append(gen_frames)
                    real_videos.append(gt_frames)

        if gen_videos and real_videos:
            fvd_score = compute_fvd(gen_videos, real_videos, feature_extractor, device)
            results["fvd"].append(fvd_score)

    averages = {}
    for metric, scores in results.items():
        if scores:
            averages[metric] = float(np.mean(scores))

    # Calculate weighted average metric score robustly
    fvd_norm = 0.0
    if results.get("fvd"):
        fvd_norm = min(results["fvd"][0] / 1000.0, 1.0)

    scores_to_avg = []
    if averages.get("psnr"):
        scores_to_avg.append(averages["psnr"] / 50.0)
    if averages.get("ssim"):
        scores_to_avg.append(averages["ssim"])
    if averages.get("lpips") is not None:
        scores_to_avg.append(1.0 - averages["lpips"])
    if averages.get("clip_s"):
        scores_to_avg.append(averages["clip_s"] / 0.3)
    if results.get("fvd"):
        scores_to_avg.append(1.0 - fvd_norm)

    if scores_to_avg:
        averages["average"] = sum(scores_to_avg) / len(scores_to_avg)

    output_file = os.path.join(output_dir, "eval_results_sota.json")
    with open(output_file, "w") as f:
        json.dump({
            "averages": averages,
            "per_sample": per_sample_results,
            "num_samples": len(per_sample_results),
        }, f, indent=2)

    print(f"\nEvaluation Results ({len(per_sample_results)} samples):")
    for metric, score in averages.items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nResults saved to {output_file}")

    return averages


def _get_feature_extractor(device):
    """Get a simple feature extractor for FVD."""
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        model.fc = torch.nn.Identity()
        model.eval()
        return model
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="SOTA Enhanced Evaluation")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing generated samples")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest for captions")
    parser.add_argument("--no_lpips", action="store_true",
                        help="Disable LPIPS computation")
    parser.add_argument("--no_clip", action="store_true",
                        help="Disable CLIP-S computation")
    parser.add_argument("--no_fvd", action="store_true",
                        help="Disable FVD computation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    args = parser.parse_args()

    evaluate_metrics(
        args.output_dir,
        args.manifest,
        use_lpips=not args.no_lpips,
        use_clip=not args.no_clip,
        use_fvd=not args.no_fvd,
        device=args.device,
    )


if __name__ == "__main__":
    main()
