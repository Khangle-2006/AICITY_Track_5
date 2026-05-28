# WTS Video Forecasting Pipeline - Tóm Tắt Toàn Diện (2026-05-26)

## 0. Tóm Tắt Baseline Hiện Tại

### Status (2026-05-26) ⭐ UPDATED
| Pipeline | Status | Base Model | Output Res | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-S ↑ | Notes |
|----------|--------|-----------|-----------|---------|---------|-----------|-----------|---------|
| **Wan2.1 SOTA** 🌟 | ✅ **PRODUCTION** | Wan2.1 Flow-DiT 1.3B | 1280×720 | **26.89 dB** | **0.947** | **0.072** | **0.253** | Main submission baseline |
| **Wan2.2** | ⚠️ Experimental | Wan2.2 Flow-DiT | 1280×720 | 20.81 dB | 0.900 | 0.119 | 0.250 | Slightly lower performance |
| **SOTA** (RectifiedFlowDiT) | ⏳ R&D | Flow-matching DiT 1.2B | 1280×720 | 17.44 dB | 0.471 | 0.823 | N/A | Reference only |
| **V2** (SDMotionUNet) | ✅ Stable (Legacy) | SD 1.5 + AnimateDiff | 1280×720 | 10.69 dB | 0.279 | 0.745 | 20.91 | Legacy baseline |

---

## 1. Tổng Quan Chung

### 1.1 Mục Tiêu
Dự đoán các frame video trong tương lai (86-94 frames) dựa trên:
- **History Frames** (16 frames): Lịch sử video đầu vào (1280×720)
- **Text Captions**: Mô tả ngữ cảnh giao thông, tương tác, điều kiện thời tiết
- **Phase Labels** (0-4): Giai đoạn nhận thức (pre-recognition → recognition → judgment → action → avoidance)
- **Sample Type**: Loại cảnh (BDD front, WTS vehicle, WTS overhead)

### 1.2 Các Metric Đánh Giá
| Metric | Hướng | Ý Nghĩa |
|--------|--------|---------|
| **PSNR** ↑ | Higher | Chất lượng tái tạo pixel-level (dB) |
| **SSIM** ↑ | Higher | Độ tương đồng cấu trúc (0-1) |
| **LPIPS** ↓ | Lower | Khoảng cách cảm nhận VGG (0-1) |
| **CLIP-S** ↑ | Higher | Alignment ngữ nghĩa giữa frames & captions (0-1) |
| **FVD** ↓ | Lower | Fréchet Video Distance - chất lượng temporal (I3D Kinetics-400) |

### 1.3 Độ Phân Giải Dataset
- **Input (History)**: 1280 × 720 pixels (16 frames)
- **Output (Predicted)**: 1280 × 720 pixels (86-94 frames, dynamic)
- **Native Training Res**: 480 × 832 pixels (Wan2.1 native)
- **Internal Processing**: 480 × 832 latents → 1280 × 720 bilinear upscale

---

## 2. Pipeline Architecture - Chi Tiết Wan2.1 SOTA ⭐

### 2.0 Wan2.1 SOTA Pipeline (PRODUCTION - 2026-05-26)

**Đặc Điểm Chính**:
| Thuộc Tính | Giá Trị |
|------------|--------|
| **Base Model** | Wan2.1 Text-to-Video (Wan-AI) |
| **Model ID** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| **Architecture** | 3D Causal VAE + Flow-Matching DiT |
| **Số Tham Số** | 1.3B (pretrained, không finetuning) |
| **Độ Phân Giải Native** | 480 × 832 pixels |
| **Độ Phân Giải Output** | 1280 × 720 pixels (bilinear upscale) |
| **Bước Denoising** | 50 steps (Euler solver, flow-matching) |
| **Guidance Strategy** | Dynamic CFG (5.0 → 1.0, cosine decay) |
| **Chunked Generation** | Autoregressive (chunk_size=81, overlap=2 frames) |
| **Background Preservation** | Optical flow (RAFT/Farneback) + homography warping |
| **Alpha Compositing** | Blend foreground (WAN) + background (warped history) |
| **Text Encoder** | UMT5 (multilingual, 768-dim embeddings) |
| **Inference Speed** | ~40-50s per 100-frame video (A6000 GPU) |
| **VRAM Required** | 12-15 GB (phù hợp A6000 24GB) |

**Measured Validation Performance** (5 samples):
```
PSNR:  26.8876 dB ✅ Excellent pixel fidelity
SSIM:  0.9470    ✅ Strong structural preservation
LPIPS: 0.0723    ✅ Low perceptual distance
CLIP-S: 0.2526   ✅ Semantic alignment
FVD:   197.1279  ✅ Good temporal distribution (3D I3D)
Average Score: 0.8115
```

### 2.1 Data Flow - Chi Tiết Luồng Xử Lý

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA LOADING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Manifest (test.jsonl):                                   │
│    {                                                            │
│      "scenario_name": "20230707_14_CN16_T1_Camera2_3",           │
│      "history_frames_dir": "data/frames/train/sample_0/",        │
│      "merged_caption": "Car approaching intersection...",        │
│      "phase_label": 2,  // 0-4 (pre-rec → avoidance)             │
│      "sample_type": "wts_vehicle",  // or bdd_front, overhead    │
│      "frame_length": 90  // Target frames (dynamic 86-94)        │
│    }                                                            │
│                                                                 │
│  Files Loaded:                                                  │
│    • 16 PNG history frames (0.png → 15.png) @ 1280×720           │
│    • Concatenated caption (56-word limit, domain-aware)          │
│    • Phase label encoded (0-4 → phase description)               │
│                                                                 │
│  Output: Raw history_imgs, caption_string, target_length        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: PREPROCESSING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Text Processing:                                               │
│    1. Build WAN-aware prompt:                                   │
│       └─ Inject phase label (pre-recognition, recognition...)   │
│    2. Tokenize (BPE, max 77 tokens):                            │
│       └─ Pad/truncate to fixed length                           │
│    3. UMT5 Encode:                                              │
│       └─ Text embedding: [1, 77, 768]                           │
│                                                                 │
│  Image Processing:                                              │
│    1. Resize history 1280×720 → 480×832:                        │
│       └─ Bilinear interpolation (maintain aspect via padding)   │
│    2. Stack 16 frames: [1, 16, 3, 480, 832]                     │
│    3. Normalize [-1, 1]: x = 2*(x/255) - 1                      │
│                                                                 │
│  Output: text_embedding, normalized_history                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 3: VAE ENCODING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: History frames [1, 16, 3, 480, 832]                     │
│  ↓                                                              │
│  3D Causal VAE Encoder:                                         │
│    └─ Conv3D (stride 2, 4x downsampling spatial)                │
│    └─ Residual + Attention blocks                               │
│    └─ Sample from qφ(z|x) Gaussian distribution                 │
│  ↓                                                              │
│  Output: Latent tensor [1, 16, 4, 120, 208]                     │
│          (Compression: 4x spatial, 1x temporal, 4 channels)     │
│                                                                 │
│  Memory saved: 1280×720×3 → 120×208×4 (48× compression)         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│            PHASE 4: FLOW-MATCHING DIFFUSION (50 steps)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each denoising step t in [1000, 999, ..., 1]:               │
│                                                                  │
│    1. Classifier-Free Guidance:                                 │
│       ├─ ε_uncond = WanTransformer3D(z_t, t, empty_text)         │
│       ├─ ε_cond = WanTransformer3D(z_t, t, text_embedding)       │
│       └─ ε_pred = ε_uncond + w * (ε_cond - ε_uncond)             │
│          where w = max(1.0, 5.0 * cos(π*t/1000))                 │
│          (Cosine decay from 5.0 → 1.0)                           │
│                                                                  │
│    2. Flow-Matching Update (Euler solver):                       │
│       └─ z_{t-1} = z_t - σ_t * ε_pred                            │
│          where σ_t is flow schedule                              │
│                                                                  │
│    3. Output: Denoised latent z_0 with alpha mask                │
│       └─ Shape: [1, T, 4, 120, 208] (T = 81 per chunk)           │
│                                                                  │
│  Result: Generated future frames in latent space                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│           PHASE 5: VAE DECODING → RGB FRAMES                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Denoised latents [1, T, 4, 120, 208]                     │
│  ↓                                                               │
│  3D Causal VAE Decoder:                                          │
│    └─ Transpose Conv (upsampling 4×)                             │
│    └─ Residual + Attention blocks                                │
│    └─ Conv3D → RGB output                                        │
│  ↓                                                               │
│  Output: RGB frames [1, T, 3, 480, 832]                          │
│          (T = 81 per chunk, full range [-1, 1])                  │
│  ↓                                                               │
│  Denormalization: x_rgb = (x + 1) / 2 * 255  (0-255 range)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│     PHASE 6: OPTICAL FLOW BACKGROUND PRESERVATION                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Motivation: 70-80% pixels are static/slowly-changing background │
│              → Preserve to improve PSNR/SSIM                     │
│                                                                  │
│  Process:                                                        │
│    1. Extract optical flow from history frames:                  │
│       └─ RAFT or Farneback (multi-frame flow averaging)          │
│    2. Estimate homography transformation:                        │
│       └─ RANSAC (4-point homography from sparse matches)         │
│    3. Warp background frames deterministically:                  │
│       └─ Warped_bg = warp(history_frame, homography)             │
│    4. Extract motion mask:                                       │
│       └─ mask = binary(motion_magnitude > threshold)             │
│                                                                  │
│  Output: warped_background [1, T, 3, 480, 832]                   │
│          + motion_mask [1, T, 1, 480, 832]                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│          PHASE 7: ALPHA COMPOSITING (Foreground + Background)    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Blend Strategy:                                                 │
│    • Foreground: Generated WAN frames (moving objects)           │
│    • Background: Warped history (static scene)                   │
│    • Alpha mask: Generated by WAN as extra channel               │
│                                                                  │
│  Blending Formula:                                               │
│    output = α × foreground + (1 - α) × background                │
│    └─ α initialized from motion_mask or alpha channel            │
│    └─ Per-pixel blending for smooth transitions                  │
│                                                                  │
│  Result: Final composited frames [1, T, 3, 480, 832]             │
│          with improved background stability                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│           PHASE 8: OUTPUT RESIZING & FINALIZATION                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Composited frames [1, T, 3, 480, 832]                    │
│  ↓                                                               │
│  Bilinear Upsampling:                                            │
│    480×832 → 1280×720 (submission resolution)                    │
│  ↓                                                               │
│  Clamp Values:                                                   │
│    x = clip(x, 0, 255)  # Ensure valid uint8 range               │
│  ↓                                                               │
│  Convert to uint8 & Save as PNG:                                 │
│    for each frame in chunk:                                      │
│      save_png(frame, output_dir/sample_N/frame_id.png)           │
│  ↓                                                               │
│  Output: Final PNG frames @ 1280×720                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│       PHASE 9: AUTOREGRESSIVE CHUNKED GENERATION (if needed)     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Problem: Target frame_length can be 86-94, but chunk only       │
│           generates 81 frames per iteration                      │
│                                                                  │
│  Solution: Autoregressive generation with history sliding        │
│                                                                  │
│  Loop:                                                           │
│    total_frames = 0                                              │
│    while total_frames < target_frame_length:                     │
│                                                                  │
│      1. Repeat PHASE 4-8 (generate 81-frame chunk)               │
│      2. Append chunk to output                                   │
│      3. Update history_images:                                   │
│         └─ history_new = history_old[-2:] + generated_frames[-79:]│
│         └─ (Keep last 2 overlap frames for continuity)           │
│      4. total_frames += 79 (81 - 2 overlap)                      │
│                                                                  │
│  Result: Seamless long-video generation (86-94 frames)           │
│          via chunked diffusion with history context              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu Trúc Codebase Chính

### 3.1 Inference Scripts

| Script | Chức Năng | Hỗ Trợ |
|--------|----------|--------|
| `scripts/infer/infer_wan.py` | Wan2.1 inference pipeline | ✅ Main production script |
| `scripts/infer/infer_wan2_2.py` | Wan2.2 inference (experimental) | ⚠️ Legacy variant |
| `scripts/eval/eval_sota.py` | Compute PSNR, SSIM, LPIPS, CLIP-S, FVD | ✅ Standard evaluator |

### 3.2 Pipeline Modules

| Module | Đường Dẫn | Chức Năng |
|--------|---------|----------|
| **Wan Video Pipeline** | `src/pipelines/wan_video_pipeline.py` | Core diffusion + compositing |
| **Wan2.2 V2V Pipeline** | `src/pipelines/wan2_2_v2v_pipeline.py` | Wan2.2 variant (v2v mode) |
| **Model Wrapper** | `src/models/wrappers/forecasting_model.py` | Unified model interface |
| **Text Encoder** | `src/models/text_encoders.py` | UMT5 + CLIP encoding |
| **Compositing** | `src/models/compositing/` | Background preservation & alpha blending |

### 3.3 Data & Manifests

| File | Chứa |
|------|------|
| `data/manifests/train.jsonl` | Training set (8000+ samples) |
| `data/manifests/val.jsonl` | Validation set (~500 samples) |
| `data/manifests/val_sota.jsonl` | Validation subset for SOTA testing |
| `data/manifests/test.jsonl` | Official test set (71 samples, no GT) |
| `data/manifests/test_sota.jsonl` | SOTA test variant |
| `data/frames/train/sample_N/` | History + ground truth frames |
| `data/frames/val/sample_N/` | Validation frames |

### 3.4 Output & Checkpoints

| Path | Chứa |
|------|------|
| `prediction_wan_sota/` | Latest Wan2.1 test predictions (71 samples) |
| `prediction_wan2.1_val_5/` | Wan2.1 validation (5 samples with metrics) |
| `prediction_wan2.2_val_5/` | Wan2.2 validation (4 samples with metrics) |
| `eval_results_sota.json` | Detailed per-sample metrics |

---

## 4. Hướng Dẫn Sử Dụng - Wan2.1 Production Pipeline

### 4.1 Setup Môi Trường

```bash
# 1. Set Python path (MANDATORY)
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH

# 2. Choose free GPU
export CUDA_VISIBLE_DEVICES=0

# 3. CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Verify setup
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

### 4.2 Inference - Full Test Set (71 Samples)

```bash
# Method 1: Shell Script (Recommended)
./run_wan_inference.sh 0 Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  data/manifests/test.jsonl prediction_wan_sota

# Method 2: Python Script (Full control)
python3 scripts/infer/infer_wan.py \
  --manifest data/manifests/test.jsonl \
  --output_dir prediction_wan_sota \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 --min_guidance 1.0 \
  --strength 0.7 --flow_shift 3.0 \
  --chunk_size 81 --overlap_frames 2 \
  --h 16 \
  --output_width 1280 --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --use_step_caching
```

### 4.3 Inference - Validation Set (Quick Test - 5 Samples)

```bash
# Quick validation run (5 samples)
python3 scripts/infer/infer_wan.py \
  --manifest data/manifests/val_sota.jsonl \
  --output_dir prediction_wan_val_test \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 \
  --num_inference_steps 50 \
  --output_width 1280 --output_height 720 \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --max_samples 5
```

### 4.4 Evaluation - Compute Metrics

```bash
# Compute full metrics on validation
python3 scripts/eval/eval_sota.py \
  --output_dir prediction_wan_val_test \
  --manifest data/manifests/val_sota.jsonl \
  --device cuda

# Output: eval_results_sota.json with per-sample & aggregated metrics
```

---

## 5. Performance Comparison

### 5.1 Measured Validation Results (Official - 2026-05-26)

| Pipeline | Samples | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-S ↑ | FVD ↓ | Avg Score |
|----------|---------|--------|--------|---------|----------|-------|-----------|
| **Wan2.1 SOTA** 🌟 | 5 | **26.8876** | **0.9470** | **0.0723** | **0.2526** | **197.1** | **0.8115** |
| **Wan2.2** (Experimental) | 4 | 20.8104 | 0.9002 | 0.1192 | 0.2495 | 517.3 | 0.7023 |
| **SOTA (RectifiedFlowDiT)** | 5 | 17.4393 | 0.4710 | 0.8233 | N/A | 206.9 | N/A |
| **V2 (SD+AnimateDiff)** (Old) | 30 | 10.6895 | 0.2793 | 0.7449 | 20.9064 | N/A | N/A |

**Nhận Xét**:
- ✅ Wan2.1 vượt trội trên tất cả metrics
- ✅ PSNR cải thiện +2.5x so với V2
- ✅ SSIM cải thiện +3.4x
- ✅ LPIPS cải thiện 10× (thấp hơn là tốt)
- ⚠️ Wan2.2 experimental - không khuyến nghị production
- 📊 FVD giờ dùng I3D Kinetics-400 (stable, so sánh được)

### 5.2 Inference Speed Benchmark

| GPU | Model | Frames | Time | Speed |
|-----|-------|--------|------|-------|
| A6000 | Wan2.1 | 81 | ~45s | 1.8 fps |
| A6000 | Wan2.1 | 100 (2 chunks) | ~90s | 1.1 fps |
| A6000 | V2 (legacy) | 81 | ~120s | 0.67 fps |

**Lưu Ý**: Thời gian bao gồm loading, encoding, 50-step denoising, decoding, compositing.

---

## 6. Thông Số Cấu Hình Chi Tiết

### 6.1 Wan2.1 Pipeline Config

```python
# Model & Resolution
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
height = 480                    # Native height
width = 832                     # Native width (16:9 × 0.6)

# Denoising & Scheduling
num_inference_steps = 50        # Diffusion steps
guidance_scale = 5.0            # Initial CFG strength
min_guidance = 1.0              # Min CFG (end of schedule)
strength = 0.7                  # Video2Video edit strength
flow_shift = 3.0                # Flow scheduler shift (tuned for 480p)

# Autoregressive Chunking
chunk_size = 81                 # Frames per chunk (AnimateDiff limit ~32 → WAN native)
overlap_frames = 2              # History overlap for smoothing
num_history_frames = 16         # Context frames from input

# Output
output_height = 720
output_width = 1280

# Memory Optimization
use_fp16 = True
use_step_caching = True

# Features
use_background_preservation = True      # Optical flow + homography
use_alpha_compositing = True            # Foreground + background blending
use_dynamic_cfg = True                  # Cosine-decay CFG schedule
```

### 6.2 Hyperparameter Tuning (Optional)

| Parameter | Default | Min | Max | Effect |
|-----------|---------|-----|-----|--------|
| `guidance_scale` | 5.0 | 3.0 | 10.0 | Higher = stronger semantic control (risk: artifacts) |
| `num_inference_steps` | 50 | 30 | 100 | More steps = higher quality (cost: +20s per 10 steps) |
| `strength` | 0.7 | 0.5 | 0.9 | Lower = preserve history better |
| `chunk_size` | 81 | 64 | 100 | Larger = fewer chunks (cost: memory) |
| `overlap_frames` | 2 | 1 | 4 | Higher = smoother transitions (cost: +time) |

---

## 7. Tối Ưu Hóa VRAM - Quan Trọng!

### 7.1 VRAM Budget (A6000 24GB)

| Component | Size | % |
|-----------|------|-----|
| Model weights (bfloat16) | ~2.5 GB | 10% |
| Activations (forward pass) | ~4 GB | 17% |
| Batch buffers (history, latents) | ~3 GB | 13% |
| Optional features (flow, compositing) | ~2 GB | 8% |
| **Total Available** | **~12-15 GB** | **50-62%** |
| **Reserved (System)** | ~9 GB | 38% |

### 7.2 Memory Optimization Techniques

| Kỹ Thuật | Tiết Kiệm | Cài Đặt |
|---------|----------|--------|
| **Gradient Checkpointing** | ~2GB | Enabled by default in VAE/DiT |
| **Bfloat16 Mixed Precision** | ~1.5GB | `use_fp16=True` |
| **Step Caching** | ~0.5GB | `use_step_caching=True` |
| **Chunked VAE** | ~0.5GB | Auto-enabled for large tensors |
| **Reduced Chunk Size** | Variable | `chunk_size=64` (from 81) |

### 7.3 Environment Setup

```bash
# MANDATORY for Wan2.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Optional: verbose logging
export TORCH_CPP_LOG_LEVEL=1
```

---

## 8. Troubleshooting

### 8.1 Out-Of-Memory (OOM) Error

```bash
# Solution (in order of aggressiveness):
1. export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Try first

2. Reduce chunk_size:
   --chunk_size 64  (from 81)

3. Reduce height/width (not recommended):
   --height 384 --width 640  (from 480x832)

4. Use FP32 instead (slower, more stable):
   # Remove use_fp16=True from config

5. Switch GPU: export CUDA_VISIBLE_DEVICES=1 (find free GPU)
```

### 8.2 Low PSNR/SSIM Output

```bash
# Debugging:
1. Verify checkpoint loaded: --verbose
2. Check ground truth exists (if validation)
3. Increase denoising steps: --num_inference_steps 100
4. Increase guidance: --guidance_scale 7.0
5. Verify input normalization (0-1 range)
```

### 8.3 Temporal Flickering or Discontinuity

```bash
# Solutions:
1. Increase overlap: --overlap_frames 4 (from 2)
2. Use EMA weights (if available)
3. Increase guidance_scale: 5.0 → 7.0
4. Check history frames are consecutive
```

---

## 9. Command Reference (Cheat Sheet)

### Setup (ALWAYS)
```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Quick Test (1 sample)
```bash
python3 scripts/infer/infer_wan.py --manifest data/manifests/val_sota.jsonl \
  --output_dir /tmp/test_wan --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 --output_width 1280 --output_height 720 \
  --max_samples 1
```

### Full Validation (5-10 samples)
```bash
python3 scripts/infer/infer_wan.py --manifest data/manifests/val_sota.jsonl \
  --output_dir prediction_wan_val --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 --output_width 1280 --output_height 720 \
  --use_background_preservation --use_alpha_compositing --use_dynamic_cfg \
  --max_samples 10
```

### Full Test (71 samples - Production)
```bash
./run_wan_inference.sh 0 Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  data/manifests/test.jsonl prediction_wan_sota
```

### Evaluate (After inference)
```bash
python3 scripts/eval/eval_sota.py --output_dir prediction_wan_val \
  --manifest data/manifests/val_sota.jsonl --device cuda
```

---

## 10. Next Steps & Future Work

### 10.1 Immediate (Current)
- ✅ **PRODUCTION**: Use Wan2.1 for official submission
- ✅ **MONITORING**: Track evaluation metrics on final test set
- ⚠️ **OPTIONAL**: A/B test Wan2.2 on hold-out subset

### 10.2 Short Term (1-2 weeks)
- [ ] Explore Wan2.2 with optimized hyperparameters
- [ ] Test with larger GPUs (H100 for higher resolution)
- [ ] Ensemble Wan2.1 + SOTA for robustness
- [ ] Profile VRAM usage to identify bottlenecks

### 10.3 Medium Term (4+ weeks)
- [ ] Finetune Wan2.1 on WTS domain (if allowed by competition)
- [ ] Test flow-shift & guidance_scale ranges systematically
- [ ] Implement custom background preservation module
- [ ] Evaluate Wan2.2 14B variant (if 40GB GPU available)

---

## 11. Appendix: Legacy Baselines (Reference Only)

### A.1 V2 (SD 1.5 + AnimateDiff) - Deprecated ❌

| Thuộc Tính | Giá Trị |
|------------|--------|
| Base | SD 1.5 UNet + AnimateDiff |
| Parameters | ~1.3B |
| Resolution | 256×144 → 1280×720 (upscaled) |
| PSNR | ~10.7 dB (measured on 30 samples) |
| SSIM | ~0.28 |
| Status | Legacy - do not use for submission |

**Why Deprecated**: Much lower metrics, obsoleted by Wan2.1

### A.2 SOTA (RectifiedFlowDiT) - R&D Only ⏳

| Thuộc Tính | Giá Trị |
|------------|--------|
| Base | Flow-matching DiT 1.2B |
| Training Stages | 3 (foundation → sim2real → DPO) |
| Status | Training scripts ready, inference incomplete |
| PSNR | 17.4 dB (measured on 5 samples) |
| Recommendation | Keep for research/ablation only |

**Current Status**: Not ready for production

---

## 12. File Locations (Quick Reference)

| Item | Path |
|------|------|
| **Wan2.1 Inference** | `scripts/infer/infer_wan.py` |
| **Wan2.1 Pipeline** | `src/pipelines/wan_video_pipeline.py` |
| **Evaluator** | `scripts/eval/eval_sota.py` |
| **Test Manifest** | `data/manifests/test.jsonl` |
| **Validation Manifest** | `data/manifests/val_sota.jsonl` |
| **Output (Test)** | `prediction_wan_sota/` |
| **Output (Val)** | `prediction_wan2.1_val_5/` (reference) |
| **Metrics** | `prediction_wan2.1_val_5/eval_results_sota.json` |

---

**Last Updated**: 2026-05-26  
**Status**: ✅ Production Ready (Wan2.1 SOTA)  
**Maintainer**: WTS Baseline Team
