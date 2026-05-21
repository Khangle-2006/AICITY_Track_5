# WTS Baseline — AGENTS.md

## Two pipelines coexist

| Pipeline | Base | Scripts prefix | Shell entrypoints |
|----------|------|----------------|-------------------|
| **V2** (SD UNet) | SD 1.5 UNet + AnimateDiff | `scripts/train/run_train_v2.py`, `scripts/infer/infer.py`, `scripts/eval/compute_metrics.py` | `run_training_v2.sh`, `run_inference.sh`, `run_validation_eval.sh` |
| **SOTA** (RectifiedFlowDiT) | Flow-matching DiT + dual conditioning | `scripts/train/run_sota_train.py`, `scripts/infer/infer_sota.py`, `scripts/eval/eval_sota.py` | `run_sota_{prepare,training,inference,eval,full_pipeline}.sh` |

SOTA has 3 training stages (foundation → sim2real → DPO). `run_sota_full_pipeline.sh` runs all stages end-to-end plus inference + eval.

## Required setup before every command

```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0          # pick a free GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Key commands

```bash
# V2 training (100 epochs, 256x144, batch=1, grad_accum=4)
./run_training_v2.sh

# V2 inference (test manifest, autoregressive chunked generation)
CUDA_VISIBLE_DEVICES=2 python3 scripts/infer/infer.py \
  --manifest data/manifests/test.jsonl \
  --checkpoint checkpoints_v2/final.pt \
  --output_dir prediction_v2 --use_sd_unet --use_ema \
  --h 8 --t 8 --width 256 --height 144 \
  --output_width 1280 --output_height 720

# V2 evaluation (needs ground-truth _gt folders)
python3 scripts/eval/compute_metrics.py \
  --output_dir prediction_v2_val --manifest data/manifests/val.jsonl

# SOTA full pipeline (data prep → 3-stage train → inference → eval)
bash run_sota_full_pipeline.sh [gpu_id] [stage1_epochs] [stage2_epochs] [stage3_epochs]

# Quick smoke test (single sample)
python3 scripts/infer/infer.py --manifest data/manifests/val.jsonl \
  --checkpoint checkpoints_v2/best.pt --output_dir /tmp/test --use_sd_unet --max_samples 1
```

## Architecture gotchas

- **Chunked generation**: AnimateDiff positional embeddings cap at 32 frames. Inference generates t=8 frame chunks with sliding window (`history_img` updated per chunk). Frame length is per-sample dynamic (`frame_length` in test.jsonl, 51–120 frames).
- **VRAM**: A6000 ~21.9 GB free. Training resolution 256×144, h=8, batch=1. Use 8-bit AdamW, gradient checkpointing, frozen VAE/CLIP encoders (`torch.no_grad()`).
- **Competition metrics**: PSNR, SSIM, LPIPS, CLIP-S, FVD.
- **Test set**: 71 samples across 3 domains (BDD front 62%, WTS vehicle 28%, WTS overhead 10%). No ground truth; submission is 1280×720 PNG frames.
- **No test suite**. Validate with `--max_samples 1` inference + metric run.

## Data layout

- Manifests: `data/manifests/{train,val,test}*.jsonl`
- Frames: `data/frames/{train,val}/sample_{N}/` with `_gt/` subdir for ground truth
- Test: `frame_length` field in manifest (not fixed N)
- Phase labels: 0=pre-recognition, 1=recognition, 2=judgment, 3=action, 4=avoidance

## Docker

`/workspace/aicity-track5_docker/docker-compose.yml` mounts `/data` (shared data), `/root/.cache`, and `/workspace`. Data root is `/data/Track_5/WTS_dataset/WTS_dataset`.

## Wan2.1 SOTA Pipeline (new)

| Pipeline | Base | Features | Scripts |
|----------|------|----------|---------|
| **WAN SOTA** | Wan2.1 3D Causal VAE + Flow-Matching DiT | OptFlow background, alpha compositing, dynamic CFG, autoregressive chunking | `scripts/infer/infer_wan.py`, `src/pipelines/wan_video_pipeline.py` |

```bash
# Quick test (3 samples)
CUDA_VISIBLE_DEVICES=0 ./run_wan_inference.sh 0 Wan-AI/Wan2.1-T2V-1.3B-Diffusers data/manifests/test.jsonl prediction_wan_sota 3

# Full test set (all 71 samples)
CUDA_VISIBLE_DEVICES=0 ./run_wan_inference.sh 0 Wan-AI/Wan2.1-T2V-1.3B-Diffusers data/manifests/test.jsonl prediction_wan_sota

# Manual run with full control
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python3 scripts/infer/infer_wan.py \
  --manifest data/manifests/test.jsonl \
  --output_dir prediction_wan_sota \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 --min_guidance 1.0 \
  --strength 0.7 --flow_shift 3.0 \
  --chunk_size 81 --h 16 \
  --output_width 1280 --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --use_step_caching \
  --max_samples 3
```

### SOTA Pipeline Architecture
```
History Frames → Wan3D Causal VAE → Latents
     ↓
[UMT5 Text Encoder] → Prompts → WanTransformer3D (Rectified Flow DiT)
     ↓
Flow-Matching Denoising (50 steps, dynamic CFG)
     ↓
Wan3D Causal VAE Decode → Raw Frames
     ↓
[Optical Flow Background Preservation]
  ├─ RAFT/Farneback optical flow on history
  ├─ RANSAC homography estimation
  ├─ Motion mask extraction
  └─ Deterministic background warping
     ↓
[Adaptive Alpha Blending Compositing]
  ├─ Alpha mask from motion mask
  └─ output = α * foreground(WAN) + (1-α) * background(warped)
     ↓
Resize 480×832 → 1280×720 → Output PNGs
```

### Key Config Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | HF model ID (1.3B or 14B) |
| `height`/`width` | 480/832 | WAN native resolution |
| `num_inference_steps` | 50 | Denoising steps |
| `guidance_scale` | 5.0 | Max CFG (decays to `min_guidance`) |
| `min_guidance` | 1.0 | Min CFG at final steps |
| `strength` | 0.7 | Video2video edit strength |
| `flow_shift` | 3.0 | Scheduler flow shift (3.0 for 480p) |
| `chunk_size` | 81 | Max frames per autoregressive chunk |
| `overlap_frames` | 2 | Overlap between chunks for smoothing |
| `use_background_preservation` | True | Optical flow + homography compositing |
| `use_dynamic_cfg` | True | Cosine-decay CFG schedule |

### Per-Chunk Denoising (Dynamic CFG Mode)
When `--use_dynamic_cfg` is enabled, the pipeline directly uses `WanTransformer3DModel` and `AutoencoderKLWan` with per-step CFG scheduling instead of the canned HF pipeline. This enables the cosine-decay guidance described in the docx (Section 7.2).
