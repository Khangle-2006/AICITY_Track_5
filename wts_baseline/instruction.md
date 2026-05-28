## 1. Chiến lược Tinh chỉnh (Fine-tuning Strategy)

Để tối ưu tài nguyên VRAM và tránh hiện tượng sụp đổ hình ảnh (Catastrophic Forgetting) của mô hình gốc WAN2.2, phương pháp tối ưu nhất là sử dụng **Adapter (PEFT / LoRA)** thay vì Fine-tune toàn bộ (Full Fine-tuning).

* **Đóng băng (Freeze):** Khóa hoàn toàn 100% trọng số của các tầng Không gian (**Spatial Layers**) và bộ giải mã hình ảnh (**Wan2.2-VAE**). Việc này giữ cho WAN2.2 khả năng vẽ ô tô, người đi bộ, đường sá siêu nét và không bị học vẹt (overfitting).
* **Giải phóng (Unfreeze / Train):** Chỉ chèn LoRA hoặc huấn luyện các tham số thuộc các tầng Chú ý Thời gian (**Temporal Attention Layers**). Lớp Adapter này đóng vai trò học logic di chuyển, quỹ đạo giao thông đặc thù của tập dữ liệu WTS dựa trên văn bản mô tả (`caption_pedestrian` và `caption_vehicle`).

---

## 2. Pipeline Quy trình Thực hiện cho 81 Khung hình

Vì số frames dynamics của tập test là chuỗi rất dài đối với mô hình Diffusion, pipeline sẽ chia nhỏ thành từng đoạn ngắn (ví dụ: mỗi đoạn sinh 16 frames) và chạy cuốn chiếu:

### Bước 1: Huấn luyện Cuốn chiếu kết hợp TIC-FT Buffer
1. **Phân đoạn 1 (Frames 1-16):** * Lấy 5 frames lịch sử đầu bài cho làm `History`.
   * Tạo chuỗi khung đệm nhiễu `Buffer`.
   * Cho WAN2.2 sinh ra 16 frames tiếp theo.
2. **Phân đoạn 2 (Frames 17-32):**
   * Lấy 4-5 frames *cuối cùng vừa sinh ra* ở Phân đoạn 1 làm `History` mới.
   * Tiếp tục áp thuật toán chèn `Buffer` nhiễu tăng dần của TIC-FT để làm mịn điểm nối nối tiếp.
   * Sinh tiếp 16 frames tiếp theo.
3. **Lặp lại** cho đến khi chạm mốc frame yêu cầu của tập test.

### Bước 2: Tối ưu hóa Tràn bộ nhớ (VRAM OOM)
Do chiều dài chuỗi sinh lên tới 81 frames, bắt buộc phải bật tính năng **Temporal Tiling** trong mã nguồn inference của WAN2.2 (chia nhỏ video theo trục thời gian thành các chunk nhỏ hơn khi tính toán Attention và VAE decode).

---

## 3. Các lưu ý sống còn khi thi đấu AI City Challenge 2026
1. **Strict Temporal Causality (Tính nhân quả nghiêm ngặt):** Khi thực hiện suy luận (Inference) trên tập Test, tuyệt đối không được rò rỉ bất kỳ thông tin nào của tương lai vào chuỗi khung hình In-context. Mô hình chỉ được biết quá khứ và tự suy luận ra tương lai.

---

## 4. Implementation Details - TIC-FT Adapter cho WAN2.2

### 4.1 Kiến trúc LoRA Temporal Attention

**File:** `src/models/adapters/lora_temporal_adapter.py`

Implement các thành phần:
- **`LoRALinear`**: Thay thế linear layers với low-rank decomposition
  - Decompose: ΔW = BA^T (A ∈ ℝ^{d_in × r}, B ∈ ℝ^{d_out × r})
  - Scaling: output = αBA^T x, với α là LoRA scaling factor
  - Trainable params: (d_in + d_out) × r (rất nhỏ so với d_in × d_out)

- **`TemporalAttentionLoRAAdapter`**: Wrapper cho một temporal attention layer
  - Freeze base attention module
  - Apply LoRA to to_q, to_k, to_v, to_out projections
  - Enable/disable LoRA updates khi train/inference

- **`WAN2_2TemporalLoRAAdapter`**: Orchestrator cho toàn bộ transformer
  - Freeze 100% spatial layers (tên chứa "spatial" hoặc "cross_attn")
  - Freeze 100% VAE encoder/decoder
  - Auto-detect temporal layers (tên chứa "temporal" hoặc "time")
  - Apply LoRA adapters chỉ cho temporal layers
  - Utility: save/load LoRA weights (không save frozen base)

**Usage:**
```python
from src.models.adapters import apply_lora_to_pipeline

# Load pipeline
pipeline = WanVideoToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-...")

# Apply LoRA (replace transformer với adapter)
adapter = apply_lora_to_pipeline(
    pipeline,
    rank=8,           # LoRA rank
    alpha=1.0,        # Scaling
    enable_training=True
)

# Get trainable param count
print(f"Trainable: {adapter.get_trainable_parameters():,}")

# Train...

# Save (chỉ LoRA weights)
adapter.save_lora_weights("lora_checkpoint.pt")

# Load
adapter.load_lora_weights("lora_checkpoint.pt")
```

**Trainable Parameters:**
- Với LoRA rank=8, d_in/d_out ~ 4096:
  - Per layer: (4096 + 4096) × 8 = 65,536 params
  - Tổng ~20-50K trainable params (vs. 14B của model)
  - VRAM: ~100-200 MB để store LoRA weights

### 4.2 TIC-FT Buffer Manager

**File:** `src/models/adapters/tic_ft_buffer.py`

Implement:
- **`TICFTBufferManager`**: Quản lý việc chèn buffer frames
  - Noise schedule: linear hoặc cosine ramp từ 0 → max_timestep
  - Insert giữa condition frames và target frames
  - Buffer frames = seed_frame + α*noise, với α từ schedule
  - Tạo smooth transitions giữa các chunk

  Methods:
  - `get_noise_schedule()`: Tạo timestep schedule cho N buffer frames
  - `insert_buffer_frames()`: Chèn buffers giữa condition và target
  - `_create_buffer_frames()`: Generate buffer frames từ seed
  - `get_chunk_indices()`: Calculate rolling-window chunk positions
  - `blend_chunk_boundaries()`: Linear blend frames ở overlap region

- **`RollingWindowInferenceEngine`**: Orchestrate multi-chunk generation
  - Quản lý loop through chunks
  - Gọi pipeline per chunk
  - Blend overlapping frames
  - Accumulate output

**Usage:**
```python
from src.models.adapters import TICFTBufferManager, RollingWindowInferenceEngine

# Setup
tic_ft = TICFTBufferManager(
    buffer_frames=5,           # 5 buffer frames per chunk
    chunk_size=16,             # 16 generated frames per chunk
    overlap_frames=4,          # 4 frame overlap between chunks
    noise_schedule="cosine"    # or "linear"
)

# Use in inference
engine = RollingWindowInferenceEngine(
    pipeline=pipeline,
    tic_ft_manager=tic_ft,
    device=device
)

# Generate 81 frames
video = engine.generate_long_video(
    history_frames=history,    # [B, 5, C, H, W]
    prompt_embeds=prompts,
    total_frames=81,
    num_inference_steps=50,
    guidance_scale=7.5
)
# Output: [B, 86, C, H, W] (5 history + 81 generated)
```

**Chunk Layout:**
```
Chunk 0: [History (5)] → [Buffer (5)] → [Generated (16)]
Chunk 1: [Last 4 from Chunk0] + [New generated (16)]
         (overlap 4 frames, blend để smooth transition)
...
```

### 4.3 Training Script

**File:** `scripts/train/train_wan2_2_lora_ticft.py`

Example training loop:
```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python3 scripts/train/train_wan2_2_lora_ticft.py \
    --manifest data/manifests/train.jsonl \
    --output_dir checkpoints_lora \
    --epochs 10 \
    --batch_size 1 \
    --lora_rank 8 \
    --lora_alpha 1.0 \
    --use_tic_ft_buffers \
    --tic_ft_buffer_frames 5 \
    --num_inference_steps 50
```

**Key training parameters:**
- `--lora_rank`: LoRA rank (8-16 typical)
- `--lora_alpha`: Scaling (1.0 = full LoRA update)
- `--lora_dropout`: Dropout trong LoRA (0.05 recommended)
- `--use_tic_ft_buffers`: Enable TIC-FT buffer insertion
- `--tic_ft_buffer_frames`: Số buffer frames (5-10 typical)

### 4.4 Integration Points

**Inference (infer_wan2_2.py):**
```python
# Load pre-trained LoRA
from src.models.adapters import apply_lora_to_pipeline

pipeline = WanVideoToVideoPipeline.from_pretrained(model_id)
adapter = apply_lora_to_pipeline(pipeline, rank=8, enable_training=False)
adapter.load_lora_weights("checkpoints_lora/lora_epoch10.pt")

# Use TIC-FT for long video generation
from src.models.adapters import RollingWindowInferenceEngine, TICFTBufferManager

tic_ft = TICFTBufferManager(buffer_frames=5, chunk_size=16)
engine = RollingWindowInferenceEngine(pipeline, tic_ft, device)

# Generate 81 frames with smooth rolling window
output = engine.generate_long_video(
    history_frames=history,
    prompt_embeds=prompts,
    total_frames=81
)
```

### 4.5 Memory & Performance

**VRAM Usage:**
- Base WAN2.2 14B: ~28 GB (model weights)
- LoRA adapters: ~150 MB (trainable)
- Activations + cache: ~4-6 GB per batch
- **Total: ~32-34 GB (A6000 OK)**

**Speed:**
- Training: ~3-5 sec/batch (1 sample, 81 frames, 50 steps)
- Inference: ~2-3 sec/chunk × 6 chunks = ~15 sec per sample
- No architectural overhead (LoRA just adds low-rank terms)

**Benefits vs Full Fine-tune:**
- ✓ 10,000× fewer trainable parameters
- ✓ No catastrophic forgetting (frozen spatial/VAE)
- ✓ Fast inference (disable LoRA = zero overhead)
- ✓ Easy ensemble (combine multiple LoRA checkpoints)
- ✓ Stable training (small parameter updates)