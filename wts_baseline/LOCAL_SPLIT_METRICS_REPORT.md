# Local Split Metrics Report

## Evaluation Setup

This report uses the repository validation split as the local held-out test set because the official `test.jsonl` and `test_sota.jsonl` manifests do not include ground-truth future frames. Reference metrics such as PSNR, SSIM, LPIPS, and FVD require ground truth, so they can only be computed on the split that contains generated outputs plus `_gt` folders.

Measured commands run:

```bash
python3 scripts/eval/eval_sota.py \
  --output_dir prediction_sota_val \
  --manifest data/manifests/val_sota.jsonl \
  --device cpu

python3 scripts/eval/compute_metrics.py \
  --output_dir prediction_val_v2 \
  --manifest data/manifests/val.jsonl
```

WAN validation inference was not run in this pass because no `prediction_wan*` outputs exist and current GPU memory is already heavily used. The freest GPU had about 11.7 GB available, while the WAN pipeline is documented around 24 GB VRAM.

## Metric Direction

| Metric | Direction | Meaning |
|--------|-----------|---------|
| PSNR | Higher is better | Pixel reconstruction quality |
| SSIM | Higher is better | Structural/layout similarity |
| LPIPS | Lower is better | Perceptual distance |
| CLIP-S | Higher is better | Text-frame semantic alignment |
| FVD | Lower is better | Temporal/video distribution distance |

## Results

| Pipeline | Split / Samples | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-S ↑ | FVD ↓ | Source |
|----------|-----------------|-------:|-------:|--------:|---------:|------:|--------|
| Wan2.1 pipeline | Assumed local split target | 23-25 dB | 0.72-0.76 | 0.12-0.16 | 0.85-0.88 | ~15-20 | Repository WAN reports |
| SOTA RectifiedFlowDiT | Val split, 5 samples | 17.4393 | 0.4710 | 0.8233 | N/A | 206.8532 | Measured |
| V2 SD UNet + AnimateDiff | Val split, 30 samples | 10.6895 | 0.2793 | 0.7449 | 20.9064 | N/A | Measured |

## Interpretation

The measured local split results show that the current saved SOTA predictions outperform V2 on PSNR and SSIM, but V2 has lower LPIPS in its separate 30-sample result. These numbers are not a perfect apples-to-apples comparison because SOTA and V2 were evaluated from different saved output folders and sample counts.

The WAN pipeline remains the strongest expected option based on the documented target range. Its technical advantage comes from Wan2.1 temporal video priors, optical-flow background preservation, alpha compositing, dynamic guidance, and traffic-phase-aware prompts. Those design choices directly target the five metrics: preserving background improves PSNR/SSIM, constrained foreground generation improves LPIPS, prompt conditioning improves CLIP-S, and video-native generation improves FVD.

## Required Final Check

For a fair measured WAN-vs-SOTA comparison, rerun WAN on the same validation split when at least 24 GB GPU memory is free:

```bash
cd /workspace/wts_baseline
./run_wan_inference.sh 0 Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  data/manifests/val_sota.jsonl prediction_wan_val 10

python3 scripts/eval/eval_sota.py \
  --output_dir prediction_wan_val \
  --manifest data/manifests/val_sota.jsonl \
  --device cuda
```

Until that run completes, WAN scores should be treated as assumed target metrics, not final measured local scores.
