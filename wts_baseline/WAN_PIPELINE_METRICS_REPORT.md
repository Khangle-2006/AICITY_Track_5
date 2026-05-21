# Wan2.1 Pipeline Technical Report

## 1. Overview

The Wan2.1 pipeline is the most suitable production candidate for the AI City Challenge Track 5 video prediction task because it combines a strong pretrained video diffusion prior with deterministic traffic-scene preservation logic. Track 5 requires forecasting future driving frames from history frames and text context. The data contains mostly structured road scenes with strong background continuity, small foreground motion, camera movement, and phase-dependent pedestrian or vehicle behavior. Wan2.1 is well matched to this setting because it models temporal video structure directly instead of generating frames independently.

The current implementation is in:

- `scripts/infer/infer_wan.py`
- `src/pipelines/wan_video_pipeline.py`
- `run_wan_inference.sh`

It uses Wan2.1 text-to-video/video-to-video generation with autoregressive chunking, optical-flow background preservation, alpha compositing, dynamic classifier-free guidance, and final resizing to the required `1280x720` output.

## 2. Why Wan2.1 Fits This Dataset

### 2.1 Strong Temporal Prior

The WTS data is a video forecasting problem, not a single-image generation problem. Wan2.1 uses a video diffusion architecture with 3D temporal modeling, so it is better suited for motion continuity, camera movement, and frame-to-frame consistency than image-centric UNet baselines.

### 2.2 Background Preservation

Most pixels in driving videos are static or slowly changing background: road surface, buildings, lane markings, sky, and parked objects. The WAN pipeline preserves these regions using optical flow and homography-based background warping. This directly improves PSNR and SSIM because the model does not need to hallucinate stable scene content from scratch.

### 2.3 Foreground Editing Instead of Full Regeneration

The pipeline applies alpha compositing so generated motion affects mainly dynamic regions such as pedestrians, vehicles, and interaction zones. This reduces visual drift and keeps generated futures closer to the ground truth layout.

### 2.4 Better Text and Phase Conditioning

The pipeline builds prompts from captions, traffic phase labels, and sample type. This is important because Track 5 scenes depend on semantic intent: pre-recognition, recognition, judgment, action, and avoidance phases. Better conditioning should improve CLIP-S and reduce implausible future actions.

### 2.5 Efficient Inference Strategy

Autoregressive chunking allows long video generation while staying inside GPU memory limits. The current configuration uses `chunk_size=81`, `overlap_frames=2`, bfloat16 weights, and step caching, making it practical for validation and test-set inference.

## 3. Metric Definitions

The repository evaluates five metrics:

| Metric | Direction | Meaning |
|--------|-----------|---------|
| PSNR | Higher is better | Pixel-level fidelity to ground truth |
| SSIM | Higher is better | Structural similarity and layout preservation |
| LPIPS | Lower is better | Perceptual distance; lower means more visually similar |
| CLIP-S | Higher is better | Semantic alignment between video frames and captions |
| FVD | Lower is better | Temporal/video distribution distance |

## 4. Current Score Comparison

The workspace currently contains saved SOTA validation metrics in `prediction_sota_val/eval_results_sota.json`, but it does not contain a saved WAN evaluation JSON. The WAN values below are taken from the existing repository technical reports and should be treated as documented expected/measured targets until `prediction_wan_val/eval_results_sota.json` is generated.

| Pipeline | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-S ↑ | FVD ↓ | Status |
|----------|-------:|-------:|--------:|---------:|------:|--------|
| Wan2.1 pipeline | 23-25 dB | 0.72-0.76 | 0.12-0.16 | 0.85-0.88 | ~15-20 | Documented target/expected range |
| SOTA RectifiedFlowDiT | 17.44 dB | 0.471 | 0.823 | N/A | N/A | Measured on 5 saved validation samples |
| V2 SD UNet + AnimateDiff | 10.69 dB | 0.279 | 0.745 | 20.91 | N/A | Measured on 30 saved validation samples |

Because metric implementations differ across saved result files, the fairest comparison is to rerun `scripts/eval/eval_sota.py` for both WAN and SOTA on the same validation manifest and sample count.

## 5. Interpretation

Wan2.1 is expected to outperform the current saved SOTA outputs on PSNR and SSIM because it explicitly preserves static background and camera geometry. It should also improve LPIPS because foreground generation is constrained by the original scene instead of replacing the whole frame. CLIP-S should be stronger because WAN prompts combine scene captions, phase labels, and sample type. FVD is expected to be lower because Wan2.1 has a stronger pretrained temporal prior and uses chunk overlap to reduce discontinuities.

The main tradeoff is compute cost. WAN inference is heavier than simple baselines, but the repository estimates about `40-90` seconds per sample depending on frame count and chunking. For a competition submission, this is acceptable if the metric gain is the priority.

## 6. Reproducible Evaluation Commands

Run both pipelines through the same evaluator:

```bash
cd /workspace/wts_baseline
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WAN validation inference
./run_wan_inference.sh 0 Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  data/manifests/val_sota.jsonl prediction_wan_val 10

# WAN five-metric evaluation
python3 scripts/eval/eval_sota.py \
  --output_dir prediction_wan_val \
  --manifest data/manifests/val_sota.jsonl \
  --device cuda

# SOTA five-metric evaluation on the same manifest
python3 scripts/eval/eval_sota.py \
  --output_dir prediction_sota_val \
  --manifest data/manifests/val_sota.jsonl \
  --device cuda
```

## 7. Recommendation

Use the Wan2.1 pipeline as the main submission pipeline if the final measured validation scores match the documented range. It is technically better aligned with Track 5 because it preserves static scene structure, models temporal motion directly, uses semantic traffic-phase conditioning, and produces competition-resolution outputs with controllable memory usage. Keep the SOTA RectifiedFlowDiT pipeline as a research baseline or ablation target unless a full five-metric rerun shows it closing the gap.
