# AI City Challenge 2026: WTS Traffic Video Forecasting Context

This document serves as the primary context hand-off for any new model agent or developer taking over the WTS Traffic Video Forecasting pipeline.

---

## 1. Competition Overview
* **Competition:** AI City Challenge 2026 (Track 5)
* **Task:** Predict/forecast future traffic video frames given a set of historical frames, textual descriptions of the scenario (detailing pedestrian/vehicle interactions, weather, speed, and actions), and phase labels.
* **Objective:** Generate high-fidelity, temporally coherent future frames matching the ground truth.
* **Evaluation Metrics:**
  * **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates pixel-level reconstruction quality.
  * **SSIM (Structural Similarity Index)**: Measures structural degradation.
  * **LPIPS (Learned Perceptual Image Patch Similarity)**: Evaluates perceptual realism using deep feature representations (VGG).
  * **CLIP-S (CLIP Score)**: Measures semantic alignment between generated video frames and textual descriptions.

---

## 2. Dataset Structure
Located under `/workspace/wts_baseline/data/` (with manifests under `/workspace/wts_baseline/data/manifests/`):
* **`train.jsonl` / `val.jsonl`**: Manifest files containing paths to history and future frames, scenario text descriptions (`merged_caption`), and phase classification labels.
* **`test.jsonl`**: The release test manifest. Contains historical frame paths and captions but **no ground truth future frames** (pure hidden test set). Contains a dynamic `frame_length` field indicating how many future frames must be forecast (e.g., 86 to 94 frames).
* **Frame Resolutions**: Original dataset frames are **1280 x 720** (pixels).

---

## 3. Evolutionary Architecture Progression

### V1 Baseline (TemporalUNet)
* **Architecture**: A custom 3D convolutional `TemporalUNet` (~93M parameters).
* **Features**: Simple convolutional temporal layers without pretrained spatial priors.
* **Limitations**: Lacked perceptual realism, struggled with complex textual alignment, and yielded lower PSNR/SSIM.

### V2 SOTA Upgrade (SDMotionUNet)
* **Architecture**: Stable Diffusion 1.5 UNet integrated with AnimateDiff motion modules (~1.3B total parameters).
* **Parameters**:
  * **Trainable (~497M params)**: AnimateDiff motion adapter modules, cross-attention projections (`attn2`) for textual adaptation, and the phase classification embedding layers.
  * **Frozen (~815M params)**: SD 1.5 Spatial UNet layers, VAE encoder/decoder, and CLIP text encoder.
* **Loss Function**: Stable Diffusion MSE loss combined with **min-SNR-gamma (5.0)** loss weighting to stabilize gradient signals at high timesteps.

---

## 4. Crucial Memory & VRAM Optimizations
Because the shared A6000 GPU (GPU 2) is contested and leaves only **~21.9 GB of free VRAM**, the 1.3B+ parameter `SDMotionUNet` required aggressive optimizations to prevent Out-Of-Memory (OOM) errors during training:
1. **8-bit AdamW Optimizer**: Swapped `torch.optim.AdamW` for `bitsandbytes.optim.AdamW8bit`, reducing optimizer state memory footprint from ~4GB to ~1GB.
2. **Gradient Checkpointing**: Enabled throughout the entire SD UNet backbone.
3. **Frozen Encoders**: Wrapped CLIP text encoding and VAE video latent encoding in `torch.no_grad()` to drop activation footprints.
4. **Resolution Reduction**: Reduced training resolution to `256 x 144` (from 512x288 / 1280x720) to shrink activation sizes.
5. **Reduced Sequence Length**: History sequence length `H` was reduced from 16 to 8.
6. **Allocator Adjustments**: Exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and added manual `torch.cuda.empty_cache()` calls before forward passes to eliminate VRAM fragmentation.

---

## 5. Chunked Autoregressive Inference (Crucial Fix)
AnimateDiff motion modules contain absolute positional embeddings with a maximum length of **32 frames**. However, test videos require generating **up to 94 future frames**. 

To resolve this limitation:
* **The Fix**: Implemented **autoregressive chunked generation** in `scripts/infer/infer.py`. 
* **Mechanism**: Generates future frames in sliding-window chunks of `args.t` (8 frames). After generating a chunk of 8 frames, the model decodes them, shifts them to act as the new historical input context (`history_img`), and generates the next 8 frames. This repeats recursively until the dynamic `T_future` target length is met.

---

## 6. How to Run & Verify

### Training
Start the V2 training daemon:
```bash
cd /workspace/wts_baseline
nohup ./run_training_v2.sh > train_v2.log 2>&1 &
```
Monitor the training:
```bash
tail -f train_v2.log
```

### Inference
Generate forecasts (supports dynamic length via the chunked autoregressive pipeline):
```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/wts_baseline python3 -u scripts/infer/infer.py \
  --manifest data/manifests/test.jsonl \
  --checkpoint checkpoints_v2/best.pt \
  --output_dir prediction_v2 \
  --use_sd_unet \
  --max_samples 100 \
  --denoise_steps 20 \
  --h 8 \
  --t 8 \
  --width 256 \
  --height 144
```

### Metrics Evaluation
Compute metrics against ground truth (e.g., on the validation set where ground truth exists):
```bash
python3 scripts/eval/compute_metrics.py \
  --output_dir prediction_v2_val \
  --manifest data/manifests/val.jsonl
```

---

## 7. Current Project Status & Recommendations
* **Status**: Training has completed 100 epochs successfully. The chunked inference pipeline is verified and fully functional, yielding a clean evaluation workflow.
* **Recommendation**: For final submission, run the exact same training pipeline on an isolated GPU node (e.g., A100 or H100 with 40-80GB VRAM) where you can scale resolution back to `1280x720` or `512x288` and increase historical frames `H` to `16` to secure top leaderboard rankings.
