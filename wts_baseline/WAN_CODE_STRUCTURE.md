# Code Structure: WAN2.1 vs WAN2.2 vs WAN2.2+LoRA+TIC-FT

## Directory Tree

```
wts_baseline/
│
├── 🔷 WAN2.1 Pipeline (Baseline)
│   ├── run_wan_inference.sh                      ← Entry point WAN2.1
│   ├── scripts/infer/
│   │   ├── infer_wan.py                          ← V2V inference
│   │   └── compare_wan_vs_sota.py                ← Comparison
│   └── src/pipelines/
│       └── wan_video_pipeline.py                 ← WAN2.1 core
│           ├── WanPipelineConfig
│           ├── WanVideoToVideoPipeline
│           ├── BackgroundPreservationModule
│           ├── ForegroundBackgroundCompositor
│           ├── TemporalTiling
│           └── DynamicCFGScheduling
│
├── 🔶 WAN2.2 Pipeline (Enhanced)
│   ├── run_wan2_2_inference.sh                   ← Single GPU entry
│   ├── run_parallel_wan2_2.sh                    ← Multi-GPU entry
│   ├── scripts/infer/
│   │   ├── infer_wan2_2.py                       ← V2V inference
│   │   └── encode_prompts.py                     ← Pre-compute embeddings
│   └── src/pipelines/
│       └── wan2_2_v2v_pipeline.py                ← WAN2.2 core with MoE
│           ├── SlotAttentionAdapter
│           ├── Wan2_2MoEBlendedPipeline
│           ├── Phase-Aware Routing
│           ├── Latent Flow Guidance
│           └── Expert Blending
│
├── 🟢 WAN2.2 + LoRA + TIC-FT (NEW)
│   ├── scripts/train/
│   │   └── train_wan2_2_lora_ticft.py            ← Training entry
│   ├── scripts/infer/
│   │   └── infer_wan2_2_lora_ticft.py            ← Inference with LoRA
│   └── src/models/adapters/                      ← NEW MODULE
│       ├── __init__.py
│       ├── lora_temporal_adapter.py              ← LoRA implementation
│       │   ├── LoRALinear
│       │   ├── TemporalAttentionLoRAAdapter
│       │   ├── WAN2_2TemporalLoRAAdapter
│       │   └── apply_lora_to_pipeline()
│       └── tic_ft_buffer.py                      ← TIC-FT implementation
│           ├── TICFTBufferManager
│           └── RollingWindowInferenceEngine
│
├── 📚 Shared Modules
│   ├── src/datasets/
│   │   ├── wts_dataset.py
│   │   ├── sota_dataset.py
│   │   ├── collate.py
│   │   └── transforms.py
│   ├── src/models/
│   │   ├── disentanglement/
│   │   │   └── optical_flow.py
│   │   └── compositing/
│   │       └── alpha_blending.py
│   └── scripts/eval/
│       └── eval_sota.py
│
└── 📖 Documentation
    ├── instruction.md                           ← Setup guide
    ├── AGENTS.md                                ← Pipeline reference
    ├── WAN_PIPELINE_COMPARISON.md               ← This file
    ├── WAN_CODE_STRUCTURE.md                    ← Structure guide
    ├── TIC_FT_IMPLEMENTATION_GUIDE.md            ← LoRA+TIC-FT guide
    └── TIC_FT_QUICK_REFERENCE.md                ← Quick start
```

---

## Import Tree

### WAN2.1 Path

```python
# inference entry
scripts/infer/infer_wan.py
  ├─ from src.pipelines.wan_video_pipeline import WanPipelineConfig, build_wan_prompt
  ├─ from diffusers import WanVideoToVideoPipeline
  ├─ from src.datasets.wts_dataset import WTSDataset
  ├─ from src.models.disentanglement.optical_flow import BackgroundPreservationModule
  └─ from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor

src/pipelines/wan_video_pipeline.py
  ├─ from diffusers import AutoencoderKLWan, WanTransformer3DModel
  ├─ from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
  ├─ Custom: BackgroundPreservationModule
  ├─ Custom: ForegroundBackgroundCompositor
  ├─ Custom: TemporalTiling
  └─ Custom: DynamicCFGScheduling
```

### WAN2.2 Path

```python
# inference entry
scripts/infer/infer_wan2_2.py
  ├─ from diffusers import WanVideoToVideoPipeline, AutoencoderKLWan
  ├─ from src.pipelines.wan2_2_v2v_pipeline import Wan2_2MoEBlendedPipeline
  ├─ from src.datasets.wts_dataset import WTSDataset
  ├─ from src.models.disentanglement.optical_flow import BackgroundPreservationModule
  └─ from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor

src/pipelines/wan2_2_v2v_pipeline.py
  ├─ from diffusers import AutoencoderKLWan, WanTransformer3DModel
  ├─ Custom: SlotAttentionAdapter
  ├─ Custom: Wan2_2MoEBlendedPipeline
  ├─ Custom: Phase-Aware Semantic Expert Routing
  ├─ Custom: Inference-time Latent Flow Guidance
  └─ Shared: BackgroundPreservationModule, ForegroundBackgroundCompositor

scripts/infer/encode_prompts.py
  ├─ from diffusers import WanVideoToVideoPipeline
  ├─ Pre-compute embeddings (optional efficiency)
  └─ Save to checkpoint
```

### WAN2.2 + LoRA + TIC-FT Path

```python
# training entry
scripts/train/train_wan2_2_lora_ticft.py
  ├─ from diffusers import WanVideoToVideoPipeline
  ├─ from src.models.adapters import apply_lora_to_pipeline
  ├─ from src.models.adapters import TICFTBufferManager
  ├─ from src.datasets.wts_dataset import WTSDataset
  └─ Optimizer: AdamW (LoRA params only)

# inference entry
scripts/infer/infer_wan2_2_lora_ticft.py
  ├─ from diffusers import WanVideoToVideoPipeline
  ├─ from src.models.adapters import apply_lora_to_pipeline
  ├─ from src.models.adapters import TICFTBufferManager
  ├─ from src.models.adapters import RollingWindowInferenceEngine
  ├─ Load pre-trained LoRA checkpoint
  └─ Use TIC-FT for multi-chunk generation

src/models/adapters/lora_temporal_adapter.py
  ├─ LoRALinear: Low-rank decomposition
  ├─ TemporalAttentionLoRAAdapter: Single layer adapter
  ├─ WAN2_2TemporalLoRAAdapter: Orchestrator
  ├─ apply_lora_to_pipeline(): Convenience function
  └─ Methods: save_lora_weights(), load_lora_weights()

src/models/adapters/tic_ft_buffer.py
  ├─ TICFTBufferManager: Buffer insertion & scheduling
  ├─ RollingWindowInferenceEngine: Multi-chunk orchestration
  ├─ Methods: insert_buffer_frames(), get_chunk_indices()
  └─ Blending: blend_chunk_boundaries()
```

---

## Data Flow Comparison

### WAN2.1 Inference

```
Test manifest (71 samples)
        ↓
    Dataset loader (1 sample at a time)
        ↓
    History frames (5 frames, normalized)
        ↓
    WAN2.1 pipeline inference
        ├─ Text encoding (real-time)
        ├─ VAE encode
        ├─ Diffusion denoising (50 steps)
        └─ VAE decode
        ↓
    Generated frames (16-81 frames)
        ↓
    Background preservation
    (Optical flow + homography)
        ↓
    Alpha compositing
    (foreground + warped background)
        ↓
    Resize to 1280×720
        ↓
    Save PNG frames
        ↓
    Output directory: prediction_wan_sota/
```

### WAN2.2 Inference

```
Test manifest (71 samples)
        ↓
    Dataset loader (1 sample at a time)
        ↓
    History frames (5 frames, normalized)
        ↓
    Text embeddings (pre-computed or real-time)
        ↓
    WAN2.2 pipeline inference
        ├─ Phase-aware expert routing
        ├─ MoE blending (Expert 1 + Expert 2)
        ├─ VAE encode
        ├─ Diffusion denoising (50 steps)
        ├─ Latent flow guidance (inference-time)
        └─ VAE decode
        ↓
    Generated frames
        ↓
    Background preservation (same as WAN2.1)
        ↓
    Alpha compositing (same as WAN2.1)
        ↓
    Resize to 1280×720
        ↓
    Save PNG frames
        ↓
    Output directory: prediction_wan2.2_sota/
```

### WAN2.2 + LoRA + TIC-FT Inference

```
Test manifest (71 samples)
        ↓
    Dataset loader (1 sample at a time)
        ↓
    Load base WAN2.2 pipeline
        ↓
    Load LoRA checkpoint
        ├─ Apply LoRA to temporal attention layers
        ├─ Freeze spatial layers
        └─ Freeze VAE
        ↓
    History frames (5 frames)
        ↓
    FOR each chunk (size=16, overlap=4):
        ├─ Insert TIC-FT buffer frames (5 noisy frames)
        │   └─ Noise schedule: linear or cosine
        ├─ Concatenate: [History] + [Buffer] + [Target]
        ├─ WAN2.2 MoE pipeline inference
        │   ├─ LoRA temporal attention (trainable weights)
        │   ├─ Spatial attention (frozen)
        │   └─ VAE (frozen)
        ├─ Extract generated frames
        └─ Blend with previous chunk (overlap region)
        ↓
    Aggregate all chunks (remove overlaps)
        ↓
    Background preservation
        ↓
    Alpha compositing
        ↓
    Resize to 1280×720
        ↓
    Save PNG frames
        ↓
    Output directory: prediction_wan2_2_lora/
```

### WAN2.2 + LoRA + TIC-FT Training

```
Train manifest (N samples, e.g., 100)
        ↓
    Dataset loader (batch_size=1)
        ↓
    Load base WAN2.2 pipeline
        ↓
    Apply LoRA
        ├─ Freeze spatial layers: requires_grad = False
        ├─ Freeze VAE: requires_grad = False
        ├─ Initialize LoRA adapters on temporal attention
        └─ LoRA params: requires_grad = True (~50K params)
        ↓
    Setup optimizer
        └─ AdamW(lora_params, lr=5e-5)
        ↓
    FOR each epoch:
        └─ FOR each batch:
            ├─ Sample random timestep t
            ├─ Add noise to target frames
            ├─ Forward pass through WAN2.2 + LoRA
            ├─ Predict noise
            ├─ Compute MSE loss
            ├─ Backward pass
            ├─ Optimizer.step()
            └─ Save checkpoint (LoRA weights only)
        ↓
    Save final LoRA checkpoint
        └─ checkpoints_lora/lora_epoch10.pt
```

---

## Configuration Files

### WAN2.1

```yaml
# run_wan_inference.sh
MODEL_ID: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT: 480
WIDTH: 832
NUM_INFERENCE_STEPS: 50
GUIDANCE_SCALE: 5.0
MIN_GUIDANCE: 1.5
```

### WAN2.2

```yaml
# run_wan2_2_inference.sh
MODEL_ID: "Wan-AI/Wan2.2-T2V-1.3B-Diffusers"  # or A14B
HEIGHT: 480
WIDTH: 832
NUM_INFERENCE_STEPS: 50
GUIDANCE_SCALE: 5.0
MIN_GUIDANCE: 1.5
STRENGTH: 0.7
```

### WAN2.2 + LoRA + TIC-FT

```yaml
# training config
MODEL_ID: "Wan-AI/Wan2.2-T2V-1.3B-Diffusers"
LORA_RANK: 8
LORA_ALPHA: 1.0
LORA_DROPOUT: 0.05
LEARNING_RATE: 5e-5
BATCH_SIZE: 1
EPOCHS: 10
GRADIENT_ACCUMULATION_STEPS: 4

# TIC-FT config
BUFFER_FRAMES: 5
CHUNK_SIZE: 16
OVERLAP_FRAMES: 4
NOISE_SCHEDULE: "cosine"

# inference config
NUM_INFERENCE_STEPS: 50
GUIDANCE_SCALE: 7.5
USE_TIC_FT_BUFFERS: true
```

---

## Environment Setup

### WAN2.1

```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Inference
./run_wan_inference.sh 0 "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  data/manifests/test.jsonl prediction_wan_sota 71
```

### WAN2.2

```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single GPU
./run_wan2_2_inference.sh 0 "Wan-AI/Wan2.2-T2V-1.3B-Diffusers" \
  data/manifests/test.jsonl prediction_wan2.2_sota 71

# Multi-GPU (4 GPUs)
./run_parallel_wan2_2.sh data/manifests/test.jsonl \
  prediction_wan2.2_parallel \
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
```

### WAN2.2 + LoRA + TIC-FT

```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training
python3 scripts/train/train_wan2_2_lora_ticft.py \
  --manifest data/manifests/train.jsonl \
  --output_dir checkpoints_lora \
  --epochs 10 \
  --use_tic_ft_buffers

# Inference
python3 scripts/infer/infer_wan2_2_lora_ticft.py \
  --manifest data/manifests/test.jsonl \
  --checkpoint checkpoints_lora/lora_epoch10.pt \
  --output_dir prediction_wan2_2_lora \
  --use_tic_ft_buffers
```

---

## Dependencies

### WAN2.1
- diffusers (WanVideoToVideoPipeline)
- transformers (T5Tokenizer, T5EncoderModel)
- torch
- torchvision
- opencv-python (optical flow)
- numpy, PIL

### WAN2.2
- diffusers (WanVideoToVideoPipeline, MoE support)
- transformers
- torch
- torchvision
- opencv-python
- numpy, PIL

### WAN2.2 + LoRA + TIC-FT (Additional)
- peft (optional, we implement LoRA directly)
- torch (AdamW optimizer)
- tensorboard (logging, optional)

---

## Testing Checklist

```
□ WAN2.1
  □ Run inference on val_5.jsonl
  □ Check PSNR/SSIM/LPIPS metrics
  □ Compare with baseline numbers
  □ Test with --max_samples 1 for quick smoke test

□ WAN2.2
  □ Run inference on val_5.jsonl (1.3B model)
  □ Run inference on val_5.jsonl (14B model)
  □ Compare metrics between 1.3B and 14B
  □ Test multi-GPU parallel execution
  □ Pre-compute embeddings and test

□ WAN2.2 + LoRA + TIC-FT
  □ Train for 1 epoch on small dataset (--max_samples 10)
  □ Check LoRA checkpoint saves correctly
  □ Load checkpoint and run inference
  □ Test with/without TIC-FT buffers
  □ Compare metrics with/without LoRA
  □ Test rolling-window generation
  □ Verify background preservation still works
```

---

**Created**: May 27, 2026  
**Format**: Code structure documentation  
**Usage**: Reference guide for understanding codebase organization
