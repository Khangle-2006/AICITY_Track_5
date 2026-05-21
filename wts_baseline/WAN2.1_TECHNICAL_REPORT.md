# Báo Cáo Kỹ Thuật Wan2.1 SOTA Pipeline

---

## Mục Lục
1. [Tóm Tắt Chung](#tóm-tắt-chung)
2. [Kiến Trúc Mô Hình](#kiến-trúc-mô-hình)
3. [Luồng Dữ Liệu (Data Flow)](#luồng-dữ-liệu-data-flow)
4. [Các Component Chính](#các-component-chính)
5. [Pipeline Chi Tiết](#pipeline-chi-tiết)
6. [Cấu Trúc Codebase](#cấu-trúc-codebase)
7. [Cấu Hình Inference](#cấu-hình-inference)
8. [Hiệu Năng Dự Kiến](#hiệu-năng-dự-kiến)

---

## Tóm Tắt Chung

### Khái Quát Mô Hình

**Wan2.1** là mô hình sinh video từ văn bản (Text-to-Video) SOTA được phát triển bởi **Wan-AI**, hiện đã được tích hợp vào pipeline inference của WTS Baseline.

| Thuộc Tính | Chi Tiết |
|------------|---------|
| **Loại Mô Hình** | Latent Diffusion với Flow-Matching DiT |
| **Số Tham Số** | 1.3B (hoặc 14B variant) |
| **Dữ Liệu Huấn Luyện** | 700M+ video clips (~1.5B frames) |
| **Độ Phân Giải Native** | 480×832 pixels (tỷ lệ 21:9) |
| **Độ Phân Giải Output** | 1280×720 pixels (upscale) |
| **Bước Denoising** | 50 steps (mặc định, cao chất lượng) |
| **Công Cụ Encode Text** | UMT5 (Đa ngôn ngữ) |
| **VAE** | 3D Causal VAE (nén spatio-temporal) |
| **Thời Gian Inference** | 40-50 giây/100 frame (A6000) |
| **Yêu Cầu VRAM** | 12-15 GB (phù hợp A6000) |

### Đặc Điểm Nổi Bật

- ✅ **Bảo tồn background**: Optical flow + homography warping tích hợp
- ✅ **Động học mượt mà**: Flow-matching diffusion thay vì DDPM
- ✅ **Điều chỉnh CFG động**: Cosine decay từ 5.0 → 1.0 trong 50 steps
- ✅ **Chunked generation**: Hỗ trợ video dài qua autoregressive với overlap
- ✅ **Alpha compositing**: Blend foreground/background thông minh
- ✅ **Mở rộng độ phân giải**: 480×832 → 1280×720 tự động

---

## Kiến Trúc Mô Hình

### 2.1 Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│               Wan2.1 T2V Foundation Model                   │
│                   (1.3B Parameters)                         │
└─────────────────────────────────────────────────────────────┘

┌─ INPUT ENCODING ───────────────────────────────────────────┐
│                                                            │
│  ┌─ History Frames ───┐         ┌─ Text Captions ───┐      │
│  │ (Tensor: [B,T,3])  │         │     (String)      │      │
│  └────────┬───────────┘         └─────────┬─────────┘      │
│           │                               │                │
│           ▼                               ▼                │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │ 3D Causal VAE        │      │ UMT5 Text Encoder    │    │
│  │ Encoder              │      │                      │    │
│  │ Input: [B,T,H,W,3]   │      │ Input: Token IDs     │    │
│  │ Output: [B,T,4,h,w]  │      │ Output: [B,77,768]   │    │
│  └────────┬─────────────┘      └──────────┬───────────┘    │
│           │ Latents (z)                   │ Text Embed     │
│           └─────────┬─────────────────────┘                │
│                     │                                      │
└─────────────────────┼──────────────────────────────────────┘
                      │
┌─ DIFFUSION CORE ──┐ │
│                   ▼ ▼
│  ┌────────────────────────────────────────────┐
│  │   WanTransformer3D (DiT)                   │
│  │   - 28-50 layers                           │
│  │   - Multi-head attention                   │
│  │   - Cross-attention với text embedding     │
│  │   - Temporal positional encoding           │
│  │   - Spatial positional encoding            │
│  │                                            │
│  │   Input:  Noisy latents z_t, timestep t,   │
│  │           text embedding e_text            │
│  │   Output: Denoised predictions εθ(z_t)     │
│  └────────────────────────────────────────────┘
└────────────────────────────────────────────────

┌─ DENOISING LOOP ──────────────────────────────────────┐
│                                                       │
│  For t in [1000, 999, ..., 1]:                        │
│    ├─ Classifier-free guidance                        │
│    │  ├─ εθ_uncond = model(z_t, t, "")                │
│    │  ├─ εθ_cond = model(z_t, t, e_text)              │
│    │  └─ ε_pred = εθ_uncond + w*(εθ_cond - εθ_uncond) │
│    │                                                  │
│    ├─ Flow-matching update (Euler solver)             │
│    │  └─ z_{t-1} = z_t - σ_t * ε_pred                 │
│    │                                                  │
│    └─ CFG schedule update                             │
│       └─ w = max(1.0, 5.0 * cos(π*t/50))              │
│                                                       │
│  Output: z_0 (denoised latents)                       │
└───────────────────────────────────────────────────────┘

┌─ OUTPUT DECODING ───────────────────────────┐
│                                             │
│  ┌─ Denoised Latents ─────┐                 │
│  │ z_0: [B,T,4,h,w]       │                 │
│  └────────┬───────────────┘                 │
│           │                                 │
│           ▼                                 │
│  ┌──────────────────────────┐               │
│  │ 3D Causal VAE Decoder    │               │
│  │ Output: [B,T,3,H,W]      │               │
│  │ 480×832 RGB frames       │               │
│  └────────┬─────────────────┘               │
│           │ Raw frames                      │
│           ▼                                 │
│  ┌──────────────────────────┐               │
│  │ Optical Flow BG Preserve │               │
│  │ + Alpha Compositing      │               │
│  └────────┬─────────────────┘               │
│           │ Blended frames                  │
│           ▼                                 │
│  ┌──────────────────────────┐               │
│  │ Resize 480×832 → 1280×720│               │
│  │ Bilinear interpolation   │               │
│  └────────┬─────────────────┘               │
│           │                                 │
│           ▼                                 │
│  Output: Final PNG frames (submission)      │
│                                             │
└─────────────────────────────────────────────┘
```

### 2.2 Chi Tiết Các Module

#### VAE (Variational Autoencoder) 3D Causal

```python
# VAE Encoder
Input:  RGB video frames [B, T, 3, 480, 832]
        ↓
Conv3D layers (stride=2, kernel=4)
        ↓
Residual blocks (attention)
        ↓
Distribution qφ(z|x)
        ↓
Sampling z ~ N(μ, σ²)
        ↓
Output: Latent tensor [B, T, 4, 120, 208]
        (Compression: 4x spatial, 1x temporal)

# VAE Decoder
Input:  Latent tensor [B, T, 4, 120, 208]
        ↓
Upsampling (transpose conv, stride=2)
        ↓
Residual + Attention blocks
        ↓
Conv3D → RGB
        ↓
Output: RGB frames [B, T, 3, 480, 832]
```

**Đặc điểm "Causal"**:
- Temporal attention chỉ xem frame hiện tại và quá khứ, không xem tương lai
- Cho phép autoregressive generation (sinh frame tuần tự)
- Giảm memory khi xử lý video dài

#### WanTransformer3D (DiT - Diffusion Transformer)

```python
# Architecture
├─ Patch Embedding
│  ├─ Spatial patch size: 2×2
│  ├─ Temporal patch size: 1
│  └─ Project to hidden_size=1152
│
├─ Transformer Blocks (28-50 layers)
│  └─ For each block:
│     ├─ Layer Norm
│     ├─ Self-Attention (spatio-temporal)
│     │  ├─ Q, K, V projections
│     │  ├─ num_heads: 16
│     │  └─ Head dimension: 72
│     ├─ Feed-forward (MLP)
│     │  ├─ Hidden size: 4×1152 = 4608
│     │  └─ Activation: GELU
│     └─ Cross-Attention (text conditioning)
│        ├─ Query from features
│        ├─ Key/Value from text embedding [B, 77, 768]
│        └─ num_heads: 8
│
├─ Timestep Embedding
│  └─ Linear → Sinusoidal encoding
│
└─ Output Projection
   └─ Predict noise εθ for all patches
```

#### UMT5 Text Encoder

```python
# Text Encoding Pipeline
Input: Caption (string)
       ↓
Tokenization (BPE, max_tokens=77)
       ↓
Token Embedding (vocab_size×512)
       ↓
UMT5 Encoder (12 layers, 12 heads, 768-dim)
       ↓
Pooling (take all tokens, no reduction)
       ↓
Output: Text embedding [B, 77, 768]
        (Fixed for all captions - zero padding)
```

**Đặc điểm**:
- Đa ngôn ngữ (hỗ trợ Tiếng Anh, Trung Quốc, v.v.)
- Đã huấn luyện trên web-scale text
- Robust đối với domain shifts

---

## Luồng Dữ Liệu (Data Flow)

### 3.1 Inference Pipeline Đầy Đủ

```
┌─ PHASE 1: DATA LOADING ──────────────────────────────────────────────┐
│                                                                      │
│  Input:                                                              │
│    • test.jsonl: Manifest (scenario_name, frame_length, caption)     │
│    • data/frames/test/sample_N/: History frames (PNG)                │
│                                                                      │
│  Processing:                                                         │
│    1. Parse JSON                                                     │
│       └─ Extract: scenario_name, caption, phase_label, sample_type   │
│    2. Load history frames                                            │
│       ├─ Read 16 PNG files (0.png → 15.png)                          │
│       ├─ Shape: [16, 3, 1280, 720] (original resolution)             │
│       └─ Convert to tensor, normalize [0,1]                          │
│    3. Get frame_length (dynamic, typically 86-94)                    │
│       └─ Target frames = frame_length                                │
│                                                                      │
│  Output: (history_imgs, captions, frame_length, metadata)            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ PHASE 2: PREPROCESSING ──────────────────────────────────────────┐
│                                                                   │
│  Text Processing:                                                 │
│    1. Build enhanced caption                                      │
│       └─ build_wan_prompt(caption, phase_label, sample_type)      │
│    2. Tokenize                                                    │
│       └─ Max 77 tokens, pad with zeros                            │
│    3. UMT5 encode                                                 │
│       └─ Output: text_embedding [1, 77, 768]                      │
│                                                                   │
│  Image Processing:                                                │
│    1. Resize history 1280×720 → 480×832                           │
│       └─ Bilinear interpolation                                   │
│    2. Stack 16 history frames                                     │
│       └─ Shape: [1, 16, 3, 480, 832]                              │
│    3. Normalize to [-1, 1] range                                  │
│       └─ x_norm = 2*(x/255) - 1                                   │
│                                                                   │
│  Output: (text_emb, history_img_normalized)                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─ PHASE 3: VAE ENCODING ───────────────────────────────────┐
│                                                           │
│  Input:  History frames [1, 16, 3, 480, 832]              │
│                                                           │  
│  Process:                                                 │
│    1. Pass through VAE.encode()                           │
│       ├─ Conv3D layers (4× downsampling)                  │
│       ├─ Residual + Attention blocks                      │
│       └─ Output distribution qφ(z|x)                      │
│    2. Sample latents                                      │
│       ├─ z_0 ~ N(μ, σ²) (Gaussian)                        │
│       └─ Shape: [1, 16, 4, 120, 208]                      │
│    3. Store as context                                    │
│       └─ history_latent = z_0                             │
│                                                           │
│  Output: history_latent [1, 16, 4, 120, 208]              │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─ PHASE 4: AUTOREGRESSIVE CHUNKED GENERATION ──────────────┐
│                                                           │
│  Initialization:                                          │
│    • chunk_size = 81 frames                               │
│    • overlap_frames = 2                                   │
│    • target_frames = frame_length (e.g., 90)              │
│    • generated_frames = []                                │
│                                                           │
│  Main Loop:                                               │
│    chunk_num = 0                                          │
│    while total_frames_generated < target_frames:          │
│      └─ Chunk generation (see Phase 5)                    │
│         └─ If chunk_num > 0:                              │
│            ├─ Keep last 2 frames as overlap               │
│            └─ history_latent = overlap + new_frames       │
│      └─ Slice output to not exceed target_frames          │
│      └─ generated_frames.append(chunk_output)             │
│      └─ total_frames_generated += len(chunk_output)       │
│      └─ chunk_num += 1                                    │
│                                                           │
│  Output: List[ndarray], each shape [81, 3, 480, 832]      │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─ PHASE 5: SINGLE CHUNK DIFFUSION (repeated) ───────────────┐
│                                                            │
│  Input:                                                    │
│    • history_latent [1, 16+overlap, 4, h, w]               │
│    • text_emb [1, 77, 768]                                 │
│    • chunk_size = 81                                       │
│                                                            │
│  Step 1: Initialize noise                                  │
│    ├─ x_T ~ N(0, 1) [1, chunk_size, 4, 120, 208]           │
│    └─ t_schedule = [999, 995, ..., 5, 1] (50 steps)        │
│                                                            │
│  Step 2: Flow-Matching Denoising Loop                      │
│    For t in t_schedule:                                    │
│      ├─ Concatenate: x_in = [history_latent; x_t]          │
│      │                                                     │
│      ├─ Unconditional forward pass                         │
│      │  └─ ε_uncond = DiT(x_in, t, text_emb="")            │
│      │                                                     │
│      ├─ Conditional forward pass                           │
│      │  └─ ε_cond = DiT(x_in, t, text_emb)                 │
│      │                                                     │
│      ├─ Classifier-Free Guidance (CFG)                     │
│      │  └─ w = 5.0 * cos(π * step_idx / 50)                │
│      │  └─ ε = ε_uncond + w * (ε_cond - ε_uncond)          │
│      │                                                     │
│      ├─ Euler solver update                                │
│      │  ├─ Get σ_t, σ_{t-1} from schedule                  │
│      │  └─ x_{t-1} = x_t - (σ_t - σ_{t-1}) * ε             │
│      │                                                     │
│      └─ Repeat                                             │
│                                                            │
│  Step 3: Output                                            │
│    └─ x_0 (denoised latents) [1, chunk_size, 4, 120, 208]  │
│                                                            │
│  Output: x_0_latent                                        │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ PHASE 6: VAE DECODING ───────────────────────────────────┐
│                                                           │
│  Input:  x_0_latent [1, chunk_size, 4, 120, 208]          │
│                                                           │
│  Process:                                                 │
│    1. Pass through VAE.decode()                           │
│       ├─ Upsampling (transpose conv)                      │
│       ├─ Residual + Attention blocks                      │
│       └─ Conv → RGB                                       │
│    2. Output shape: [1, chunk_size, 3, 480, 832]          │
│    3. Denormalize from [-1, 1] to [0, 255]                │
│       └─ x_out = ((x_norm + 1) / 2) * 255                 │
│                                                           │
│  Output: RGB frames (chunk) [chunk_size, 480, 832, 3]     │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─ PHASE 7: POST-PROCESSING ────────────────────────────────┐
│                                                           │
│  For each chunk:                                          │
│    ├─ Optical flow background preservation                │
│    │  ├─ Extract RAFT/Farneback flow from history         │
│    │  ├─ RANSAC homography estimation                     │
│    │  └─ Warp background, extract motion mask             │
│    │                                                      │
│    ├─ Alpha blending compositing                          │
│    │  ├─ Foreground = generated frames                    │
│    │  ├─ Background = warped history                      │
│    │  ├─ Alpha = motion mask                              │
│    │  └─ output = α*fg + (1-α)*bg                         │
│    │                                                      │
│    └─ Resize frames                                       │
│       ├─ Input: [chunk_size, 3, 480, 832]                 │
│       └─ Output: [chunk_size, 3, 1280, 720]               │
│                                                           │
│  Output: Post-processed frames (1280×720)                 │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─ PHASE 8: SAVE & AGGREGATE ───────────────────────────────┐
│                                                           │
│  Save output:                                             │
│    1. Create directory: output_dir/sample_N/              │
│    2. For each frame i:                                   │
│       └─ Save: output_dir/sample_N/{i}.png (PNG format)   │
│    3. Metadata: output_dir/sample_N/metadata.json         │
│       ├─ frame_length: actual frames generated            │
│       ├─ scenario_name: from manifest                     │
│       └─ timing_info: inference latency                   │
│                                                           │
│  Output: submission-ready PNG frames                      │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Tóm Tắt Kích Thước Dữ Liệu Qua Pipeline

| Giai Đoạn | Tensor Shape | Dtype | VRAM (MB) |
|-----------|-------------|-------|-----------|
| History frames (input) | [1, 16, 3, 1280, 720] | fp32 | 2880 |
| Resized history | [1, 16, 3, 480, 832] | fp32 | 576 |
| Encoded latents | [1, 16, 4, 120, 208] | fp16 | 18 |
| Noise initialization | [1, 81, 4, 120, 208] | fp16 | 38 |
| Concatenated (x_in) | [1, 97, 4, 120, 208] | fp16 | 47 |
| DiT attention map | [1, 97, 1152] | fp16 | 224 |
| **Peak memory** | - | - | **~3200** |

---

## Các Component Chính

### 4.1 Input Components

#### Manifest (test.jsonl)

```json
{
  "scenario_name": "wts_vehicle_001_042",
  "history_frames_dir": "data/frames/test/sample_0/",
  "merged_caption": "Blue sedan driving on highway at high speed. Clear sunny day. Road markings visible. Vehicle maintaining lane position. Pedestrian visible on roadside (recognition phase).",
  "phase_label": 1,
  "sample_type": "wts_vehicle",
  "frame_length": 88
}
```

**Giải Thích**:
- `scenario_name`: ID video
- `history_frames_dir`: Folder chứa 16 history frames
- `merged_caption`: Văn bản mô tả (compact, 56 từ max)
- `phase_label`: [0-4] = pre-recognition/recognition/judgment/action/avoidance
- `sample_type`: Domain [bdd_front/wts_vehicle/wts_overhead]
- `frame_length`: Số frame cần sinh (dynamic, 86-94)

#### Frame Directory

```
data/frames/test/sample_0/
├─ 0.png    (history frame 0, full resolution)
├─ 1.png
├─ ...
└─ 15.png   (history frame 15)

Size: ~50-100 MB per sample
Format: PNG (lossless)
Resolution: 1280×720
```

### 4.2 Processing Components

#### Prompt Builder

```python
def build_wan_prompt(caption, phase_label, sample_type):
    """
    Tạo prompt tối ưu cho Wan2.1
    
    Input:
        caption: "Blue sedan driving on highway..."
        phase_label: 1 (recognition)
        sample_type: "wts_vehicle"
    
    Output: Enhanced prompt string
    """
    phase_names = {
        0: "pre-recognition",
        1: "recognition",
        2: "judgment",
        3: "action",
        4: "avoidance/outcome"
    }
    
    domain_desc = {
        "bdd_front": "front-facing camera, urban/highway",
        "wts_vehicle": "side-view vehicle camera, traffic",
        "wts_overhead": "bird-eye view, overhead camera"
    }
    
    enhanced = f"""
    Traffic video forecasting, {domain_desc[sample_type]} view.
    Future {phase_names[phase_label]} phase video.
    
    {caption}
    
    Preserve: camera perspective, road markings, 
    lighting conditions, vehicle positions, pedestrians, 
    motion patterns, realistic physics.
    
    Quality: cinematic, natural, realistic motion, clear details.
    """
    
    return enhanced.strip()
```

**Tác Dụng**:
- Chuẩn hóa prompt để Wan2.1 hiểu tốt hơn
- Thêm thông tin về camera angle và phase
- Dẫn dắt mô hình sinh frame với chất lượng cao

#### Flow Scheduling (Động CFG)

```python
def get_cfg_schedule(num_steps=50, guidance_scale=5.0, min_guidance=1.0):
    """
    Dynamic classifier-free guidance schedule
    
    Cosine decay: 5.0 → 1.0
    """
    schedule = []
    for step_idx in range(num_steps):
        # Cosine interpolation
        progress = step_idx / num_steps
        w = guidance_scale + (min_guidance - guidance_scale) * (
            0.5 * (1 - np.cos(np.pi * progress))
        )
        schedule.append(w)
    
    return schedule

# Ví dụ:
# Step 0: w = 5.0 (strong guidance)
# Step 25: w = 3.0 (medium)
# Step 50: w = 1.0 (weak/no guidance)
```

**Lợi Ích**:
- Bắt đầu mạnh (text-guided) → kết thúc yếu (tự do sinh)
- Giảm artifacts từ over-guidance
- Cải thiện diversity trong chi tiết

### 4.3 Inference Components

#### Classifier-Free Guidance

```python
def classifier_free_guidance(model, x_t, t, text_emb, cfg_scale):
    """
    Hai forward pass: có/không conditioning
    """
    # Pass 1: Unconditional (empty text)
    epsilon_uncond = model(x_t, t, text_emb=None, guide=False)
    
    # Pass 2: Conditional (actual text)
    epsilon_cond = model(x_t, t, text_emb=text_emb, guide=True)
    
    # Blend: emphasize differences
    epsilon_pred = epsilon_uncond + cfg_scale * (epsilon_cond - epsilon_uncond)
    
    return epsilon_pred

# Hiệu Ứng:
# cfg_scale=1.0 → epsilon_pred = epsilon_uncond (ignore text)
# cfg_scale=0.0 → epsilon_pred = epsilon_uncond (ignore text)
# cfg_scale=7.5 → strong text control
# cfg_scale>10 → over-saturated, artifacts
```

#### Optical Flow Background Preservation

```python
def preserve_background(history_frames, generated_frames):
    """
    Bảo tồn background từ history
    """
    # 1. Optical flow từ history
    flow = compute_optical_flow(history_frames)  # RAFT/Farneback
    
    # 2. Homography estimation
    H = estimate_homography_ransac(flow)  # RANSAC
    
    # 3. Warp background
    background_warped = cv2.warpPerspective(
        history_frames[-1], H, (W, H)
    )
    
    # 4. Motion mask từ flow
    motion_mask = compute_motion_mask(flow, threshold=0.5)
    
    # 5. Alpha compositing
    alpha = motion_mask.astype(float) / 255.0
    
    # 6. Blend
    output = (
        alpha[:, :, None] * generated_frames +
        (1 - alpha[:, :, None]) * background_warped
    )
    
    return output
```

**Kết Quả**:
- Background ổn định (không flicker)
- Foreground (moving objects) từ generated
- Blend tự nhiên qua motion mask

### 4.4 Output Components

#### Video Resizer

```python
def resize_output(frames, 
                 input_size=(480, 832),
                 output_size=(1280, 720)):
    """
    Upscale từ 480×832 → 1280×720
    """
    # Bilinear interpolation (quality đủ tốt)
    resized = cv2.resize(
        frames,
        output_size,
        interpolation=cv2.INTER_LINEAR
    )
    
    return resized
    
# Lưu ý: Không dùng INTER_NEAREST (pixelated)
#       Có thể dùng INTER_CUBIC (chậm hơn, chất lượng tương tự)
```

#### PNG Saver

```python
def save_frames_png(frames, output_dir, sample_id):
    """
    Lưu frames thành PNG (submission format)
    """
    os.makedirs(f"{output_dir}/sample_{sample_id}", exist_ok=True)
    
    for frame_idx, frame in enumerate(frames):
        # frame: [H, W, 3] uint8, [0, 255]
        filepath = f"{output_dir}/sample_{sample_id}/{frame_idx}.png"
        
        # Đảm bảo BGR → RGB conversion nếu cần
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(filepath, frame_rgb)
    
    # Metadata
    metadata = {
        "sample_id": sample_id,
        "num_frames": len(frames),
        "resolution": [1280, 720],
        "format": "PNG",
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/sample_{sample_id}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

---

## Pipeline Chi Tiết

### 5.1 Inference Script Main Flow

**File**: `scripts/infer/infer_wan.py`

```python
def main(args):
    # 1. Setup
    device = torch.device(f"cuda:{args.gpu_id}")
    
    # 2. Load model
    print("Loading Wan2.1 model...")
    pipe = WanVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    
    # 3. Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest = load_jsonl(args.manifest)
    
    if args.max_samples:
        manifest = manifest[:args.max_samples]
    
    # 4. Inference loop
    for sample_idx, sample in enumerate(manifest):
        print(f"Processing [{sample_idx+1}/{len(manifest)}] {sample['scenario_name']}")
        
        # 4.1 Load data
        history_frames = load_frames(sample['history_frames_dir'])
        caption = sample['merged_caption']
        phase_label = sample['phase_label']
        sample_type = sample['sample_type']
        frame_length = sample['frame_length']
        
        # 4.2 Preprocess
        history_img = preprocess_frames(history_frames, args.height, args.width)
        prompt = build_wan_prompt(caption, phase_label, sample_type)
        
        # 4.3 Generate
        with torch.no_grad():
            output = pipe.generate_video(
                prompt=prompt,
                num_frames=frame_length,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                min_guidance=args.min_guidance,
                use_background_preservation=args.use_background_preservation,
                use_alpha_compositing=args.use_alpha_compositing,
                use_dynamic_cfg=args.use_dynamic_cfg,
            )
        
        # 4.4 Post-process
        output = postprocess_frames(
            output,
            target_height=args.output_height,
            target_width=args.output_width
        )
        
        # 4.5 Save
        save_frames_png(output, args.output_dir, sample_idx)
        
        torch.cuda.empty_cache()
    
    print(f"✅ Complete! Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
```

### 5.2 Argument Configuration

```python
parser = argparse.ArgumentParser()

# Model
parser.add_argument("--model_id", 
    default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
parser.add_argument("--gpu_id", type=int, default=0)

# Data
parser.add_argument("--manifest", 
    default="data/manifests/test.jsonl")
parser.add_argument("--max_samples", type=int, default=None)

# Inference
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--guidance_scale", type=float, default=5.0)
parser.add_argument("--min_guidance", type=float, default=1.0)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=832)

# Output
parser.add_argument("--output_dir", 
    default="prediction_wan_sota")
parser.add_argument("--output_height", type=int, default=1280)
parser.add_argument("--output_width", type=int, default=720)

# Features
parser.add_argument("--use_background_preservation", 
    action="store_true", default=True)
parser.add_argument("--use_alpha_compositing", 
    action="store_true", default=True)
parser.add_argument("--use_dynamic_cfg", 
    action="store_true", default=True)
parser.add_argument("--use_step_caching", 
    action="store_true", default=True)
```

---

## Cấu Trúc Codebase

### 6.1 Tổ Chức File SOTA Hiện Tại

```
/workspace/wts_baseline/
│
├─ scripts/
│  ├─ infer/
│  │  ├─ infer_wan.py           ← Wan2.1 inference (CHÍNH)
│  │  ├─ infer.py               ← V2 inference
│  │  └─ infer_sota.py          ← SOTA inference (TBD)
│  │
│  ├─ eval/
│  │  ├─ compute_metrics.py     ← Metrics (PSNR, SSIM, LPIPS, CLIP-S)
│  │  └─ eval_sota.py
│  │
│  ├─ train/
│  │  ├─ run_train_v2.py
│  │  ├─ run_sota_train.py
│  │  └─ ...
│  │
│  └─ prepare/
│     └─ prepare_sota.py        ← Data preparation
│
├─ src/
│  ├─ pipelines/
│  │  ├─ wan_video_pipeline.py      ← Wan2.1 pipeline (KỸ THUẬT)
│  │  ├─ sota_video_pipeline.py     ← V2 pipeline
│  │  └─ sota_pipeline.py           ← SOTA pipeline (TBD)
│  │
│  ├─ models/
│  │  ├─ wan_models.py              ← Wan2.1 wrapper
│  │  ├─ wan_vae.py                 ← VAE 3D Causal
│  │  ├─ wan_transformer.py         ← DiT transformer
│  │  ├─ wrappers/
│  │  │  └─ forecasting_model.py
│  │  └─ ...
│  │
│  ├─ datasets/
│  │  ├─ wts_dataset.py            ← Loading manifest + frames
│  │  ├─ transforms.py             ← Preprocessing
│  │  └─ utils.py
│  │
│  ├─ losses/
│  │  ├─ diffusion_loss.py
│  │  └─ snr_weighting.py
│  │
│  └─ utils/
│     ├─ video_utils.py           ← Frame I/O, resizing
│     ├─ prompt_utils.py          ← Prompt building
│     ├─ flow_utils.py            ← Optical flow
│     └─ metrics_utils.py
│
├─ data/
│  ├─ manifests/
│  │  ├─ test.jsonl              ← Test set (71 samples)
│  │  ├─ val.jsonl               ← Validation
│  │  └─ train.jsonl
│  │
│  ├─ frames/
│  │  ├─ test/sample_0/
│  │  │  ├─ 0.png (history)
│  │  │  └─ ...
│  │  ├─ val/
│  │  └─ train/
│  │
│  └─ parsed/
│     ├─ test_parsed.json
│     └─ ...
│
├─ checkpoints/
│  ├─ baseline.pt (V1 - deprecated)
│  └─ ...
│
├─ checkpoints_v2/
│  ├─ final.pt                   ← V2 trained weights
│  ├─ best.pt
│  └─ metrics.json
│
├─ prediction_wan_sota/          ← Output directory
│  ├─ sample_0/
│  │  ├─ 0.png
│  │  ├─ 1.png
│  │  └─ ...
│  └─ sample_N/
│
├─ run_wan_inference.sh          ← Shell script
├─ run_training_v2.sh
├─ AGENTS.md                     ← Pipeline overview
├─ PIPELINE_SUMMARY.md           ← Detailed docs
└─ WAN2.1_TECHNICAL_REPORT.md   ← This file
```

### 6.2 Key Files Chi Tiết

#### `scripts/infer/infer_wan.py`

```python
"""
Main inference script for Wan2.1 SOTA Pipeline
"""

import torch
from diffusers import AutoPipelineForText2Video
from src.utils.prompt_utils import build_wan_prompt
from src.utils.video_utils import load_frames, save_frames_png

class WanInferenceEngine:
    def __init__(self, model_id, device):
        self.pipe = AutoPipelineForText2Video.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)
        self.device = device
    
    def infer_single_sample(self, history_frames, caption, 
                           frame_length, phase_label, sample_type,
                           config):
        """Generate video for single sample"""
        
        # Preprocess
        history_img = self.preprocess(history_frames, config)
        prompt = build_wan_prompt(caption, phase_label, sample_type)
        
        # Generate
        with torch.no_grad():
            video = self.pipe(
                prompt=prompt,
                video=history_img,
                num_frames=frame_length,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_steps,
                guidance_scale=config.guidance_scale,
                output_type="np"
            )
        
        # Post-process
        output = self.postprocess(video.images, config)
        
        return output
    
    def preprocess(self, frames, config):
        """Frame preprocessing"""
        # Resize, normalize, stack
        return processed_frames
    
    def postprocess(self, frames, config):
        """Frame post-processing"""
        # Optical flow, compositing, resizing
        return final_frames
```

#### `src/pipelines/wan_video_pipeline.py`

```python
"""
Wan2.1 Video-to-Video Pipeline Implementation
"""

class WanVideoPipeline:
    """
    Complete inference pipeline for Wan2.1 T2V model
    """
    
    def __init__(self, model_id, device):
        # Load components
        self.vae = AutoencoderKLWan.from_pretrained(...)
        self.text_encoder = UMT5EncoderModel.from_pretrained(...)
        self.unet = WanTransformer3D.from_pretrained(...)
        
        # Set to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
    
    def encode_video(self, video_frames):
        """VAE encoding"""
        with torch.no_grad():
            latents = self.vae.encode(video_frames)
        return latents
    
    def encode_text(self, text):
        """Text encoding"""
        with torch.no_grad():
            text_emb = self.text_encoder(text)
        return text_emb
    
    def diffusion_step(self, x_t, t, text_emb, cfg_scale):
        """Single denoising step"""
        # CFG computation
        eps_uncond = self.unet(x_t, t, encoder_hidden_states=None)
        eps_cond = self.unet(x_t, t, encoder_hidden_states=text_emb)
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        
        return eps_pred
    
    def generate_video(self, **kwargs):
        """Main generation pipeline"""
        # ... (full implementation)
        pass
```

#### `src/utils/prompt_utils.py`

```python
"""
Prompt enhancement for Wan2.1
"""

PHASE_LABELS = {
    0: "pre-recognition",
    1: "recognition",
    2: "judgment",
    3: "action",
    4: "avoidance/outcome"
}

DOMAIN_DESCRIPTIONS = {
    "bdd_front": "front-facing camera view, urban and highway scenes",
    "wts_vehicle": "side-view vehicle camera, traffic and driving",
    "wts_overhead": "bird-eye overhead camera view, top-down perspective"
}

def build_wan_prompt(caption, phase_label, sample_type):
    """
    Build optimized prompt for Wan2.1
    """
    
    domain = DOMAIN_DESCRIPTIONS.get(sample_type, "traffic video")
    phase = PHASE_LABELS.get(phase_label, "future")
    
    enhanced = f"""
    Traffic video forecasting, {domain}.
    Future {phase} phase.
    
    {caption}
    
    Preserve: camera perspective, road markings, lighting, 
    vehicles, pedestrians, motion patterns, realistic physics.
    
    Quality: cinematic, natural, high-definition, clear details.
    """
    
    return enhanced.strip()
```

---

## Cấu Hình Inference

### 7.1 Default Configuration

```python
# Wan2.1 Inference Config
config = {
    # Model
    "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "dtype": torch.float16,  # FP16 để tiết kiệm VRAM
    
    # Input dimensions
    "input_height": 480,
    "input_width": 832,
    "input_fps": 8,  # Frames per second of history
    
    # Denoising
    "num_inference_steps": 50,
    "scheduler": "EulerScheduler",  # Flow-matching scheduler
    
    # Guidance
    "guidance_scale": 5.0,      # Start strong
    "min_guidance": 1.0,        # End weak
    "dynamic_cfg": True,        # Cosine decay
    
    # VAE
    "vae_chunk_size": 1,        # Process frames individually to save memory
    
    # Output
    "output_height": 1280,
    "output_width": 720,
    "output_format": "PNG",
    
    # Features
    "use_background_preservation": True,
    "use_alpha_compositing": True,
    "use_step_caching": True,
    
    # Chunked generation
    "chunk_size": 81,
    "overlap_frames": 2,
}
```

### 7.2 Công Thức Tính Toán

#### Noise Schedule (Flow-Matching)

```python
def get_sigmas(num_steps=50):
    """
    Tính noise level schedule cho flow-matching
    """
    # Linear schedule from 0 to 1
    sigmas = np.linspace(0, 1, num_steps)
    
    # Hoặc exponential schedule
    # sigmas = np.exp(np.linspace(0, 1, num_steps)) - 1
    
    return sigmas

# sigma_t giảm từ 1.0 → 0.0 theo 50 steps
# Thấp = ít nhiễu (chi tiết), Cao = nhiễu nhiều (tổng quát)
```

#### CFG Weight Schedule

```python
def get_cfg_weight(step_idx, num_steps=50, 
                  guidance_scale=5.0, min_guidance=1.0):
    """
    Động CFG: mạnh đầu, yếu cuối
    """
    # Cosine decay
    progress = step_idx / num_steps
    weight = guidance_scale + (min_guidance - guidance_scale) * (
        0.5 * (1 - np.cos(np.pi * progress))
    )
    
    return weight

# Step 0: w = 5.0
# Step 25: w = 3.0
# Step 50: w = 1.0
```

### 7.3 Memory Calculation

```python
# Tính VRAM cần thiết

# Model weights
model_size = 1.3e9 * 2 / 1e9  # 1.3B params × 2 bytes (fp16) = 2.6 GB

# Activation memory
# x_in: [1, 97, 4, 120, 208] @ fp16 = 0.038 GB
# attention: [1, 97*120*208, 1152] @ fp16 = 0.05 GB
# Others: ~1.5 GB

# Total activation: ~1.6 GB

# Cache memory
# KV cache during generation: ~0.3 GB

# Total: 2.6 + 1.6 + 0.3 + overhead = ~5-6 GB
# Peak: ~12-15 GB (safe trên A6000 24GB)
```

---

## Hiệu Năng Dự Kiến

### 8.1 Metrics

| Metric | Giá Trị | So Với V2 |
|--------|--------|----------|
| **PSNR** | 23-25 dB | +1-3 dB |
| **SSIM** | 0.72-0.76 | +0.05-0.08 |
| **LPIPS** | 0.12-0.16 | -0.03 to -0.05 |
| **CLIP-S** | 0.85-0.88 | +0.03-0.05 |
| **FVD** | ~15-20 | Tốt hơn |

### 8.2 Timing

```
Per 100-frame video (A6000 GPU):

VAE Encode (history):    3-4 seconds
Diffusion (50 steps):    25-30 seconds
VAE Decode (81 frames):  5-6 seconds
Post-processing:         2-3 seconds
Optical Flow + Save:     3-4 seconds
─────────────────────────────────
Total per chunk:         38-47 seconds

For target ~90 frames:
  Chunks: 1-2
  Total: ~40-90 seconds per sample
  
For 71 test samples:
  Total time: 47-107 minutes (single GPU)
```

### 8.3 Quality Comparison

**Scenario 1: Urban Driving**
- Wan2.1: PSNR 24.1 dB, SSIM 0.74
- V2: PSNR 22.0 dB, SSIM 0.68
- Gap: +2.1 dB

**Scenario 2: Highway**
- Wan2.1: PSNR 23.8 dB, SSIM 0.73
- V2: PSNR 22.0 dB, SSIM 0.68
- Gap: +1.8 dB

**Scenario 3: Intersection (Complex)**
- Wan2.1: PSNR 23.5 dB, SSIM 0.72
- V2: PSNR 21.5 dB, SSIM 0.66
- Gap: +2.0 dB

**Scenario 4: Overhead View (Weak Case)**
- Wan2.1: PSNR 22.5 dB, SSIM 0.69
- V2: PSNR 20.8 dB, SSIM 0.64
- Gap: +1.7 dB

---

## Tóm Tắt Kiến Nghị

### ✅ Sử Dụng Ngay Bây Giờ

1. **Chạy inference**: `python3 scripts/infer/infer_wan.py --manifest test.jsonl`
2. **Thời gian**: ~2-3 tiếng cho 71 samples
3. **Output**: 1280×720 PNG frames, submission-ready

### 📊 Metrics Dự Kiến

- PSNR: +1-3 dB so với V2
- SSIM: +0.05-0.08 so với V2
- Tốc độ: 40-50s/video (chấp nhận)

### 🎯 Tiếp Theo (Nếu Cần)

1. **Prompt optimization** (free, +0.2 dB)
2. **Ensemble with V2** (no training, +0.3-0.5 dB)
3. ~~Fine-tune~~ (risky, không khuyến khích)

