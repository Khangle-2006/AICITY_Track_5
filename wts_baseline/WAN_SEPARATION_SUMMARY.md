# Tách rồi: WAN2.1 vs WAN2.2 vs WAN2.2+LoRA+TIC-FT

## 📊 Tóm tắt

| | **WAN2.1** | **WAN2.2** | **WAN2.2+LoRA+TIC-FT** |
|---|----------|----------|----------------------|
| **Model** | 1.3B Rectified Flow | 1.3B/14B Flow-Matching | 14B + LoRA adapters |
| **Entry Point** | `run_wan_inference.sh` | `run_wan2_2_inference.sh` | Training/inference scripts |
| **Inference Script** | `infer_wan.py` | `infer_wan2_2.py` | `infer_wan2_2_lora_ticft.py` |
| **Pipeline Code** | `wan_video_pipeline.py` | `wan2_2_v2v_pipeline.py` | Uses WAN2.2 + adapters |
| **Adapter Code** | None | Slot Attention (disabled) | LoRA + TIC-FT adapters |
| **Training** | ❌ Not supported | ❌ Not supported | ✅ LoRA only |
| **VRAM** | ~20 GB | ~21 GB (1.3B) / ~28 GB (14B) | ~32-34 GB |
| **Speed** | ~10 sec/sample | ~12 sec/sample | ~15 sec/sample |

---

## 🗂️ File Separation

### WAN2.1 (Baseline SOTA)
```
run_wan_inference.sh
scripts/infer/
├── infer_wan.py ⭐
└── compare_wan_vs_sota.py
src/pipelines/
└── wan_video_pipeline.py ⭐
```
**Mô tả**: Baseline SOTA với background preservation

---

### WAN2.2 (Enhanced SOTA)
```
run_wan2_2_inference.sh
run_parallel_wan2_2.sh
scripts/infer/
├── infer_wan2_2.py ⭐
└── encode_prompts.py
src/pipelines/
└── wan2_2_v2v_pipeline.py ⭐
```
**Mô tả**: MoE-enhanced SOTA với expert routing

---

### WAN2.2 + LoRA + TIC-FT (NEW)
```
scripts/train/
└── train_wan2_2_lora_ticft.py ⭐ Training entry
scripts/infer/
└── infer_wan2_2_lora_ticft.py ⭐ Inference entry
src/models/adapters/ ⭐ NEW MODULE
├── __init__.py
├── lora_temporal_adapter.py ⭐
└── tic_ft_buffer.py ⭐
```
**Mô tả**: WAN2.2 + LoRA (50K trainable params) + TIC-FT buffers

---

## 🚀 Quick Commands

### WAN2.1
```bash
./run_wan_inference.sh 0 "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  data/manifests/test.jsonl prediction_wan_sota 71
```

### WAN2.2 (1.3B)
```bash
./run_wan2_2_inference.sh 0 "Wan-AI/Wan2.2-T2V-1.3B-Diffusers" \
  data/manifests/test.jsonl prediction_wan2.2_sota 71
```

### WAN2.2 (14B)
```bash
./run_parallel_wan2_2.sh data/manifests/test.jsonl prediction_wan2.2 \
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
```

### WAN2.2 + LoRA + TIC-FT (Training)
```bash
python3 scripts/train/train_wan2_2_lora_ticft.py \
  --manifest data/manifests/train.jsonl \
  --epochs 10 --use_tic_ft_buffers --lora_rank 8
```

### WAN2.2 + LoRA + TIC-FT (Inference)
```bash
python3 scripts/infer/infer_wan2_2_lora_ticft.py \
  --manifest data/manifests/test.jsonl \
  --checkpoint checkpoints_lora/lora_epoch10.pt \
  --use_tic_ft_buffers
```

---

## 📈 Features Comparison

| Feature | WAN2.1 | WAN2.2 | WAN2.2+LoRA |
|---------|--------|--------|------------|
| **Rectified Flow** | ✅ | Enhanced | Enhanced |
| **MoE Experts** | ❌ | ✅ | ✅ |
| **Phase-Aware Routing** | ❌ | ✅ | ✅ |
| **Latent Flow Guidance** | ❌ | ✅ | ✅ |
| **Background Preservation** | ✅ | ✅ | ✅ |
| **Alpha Blending** | ✅ | ✅ | ✅ |
| **LoRA Adapters** | ❌ | ❌ | ✅ |
| **TIC-FT Buffers** | ❌ | ❌ | ✅ |
| **Pre-computed Embeddings** | ❌ | ✅ | ✅ |
| **Fine-tuning Support** | ❌ | ❌ | ✅ |

---

## 📁 Module Organization

```python
# WAN2.1 imports
from src.pipelines.wan_video_pipeline import WanPipelineConfig, build_wan_prompt
from src.models.disentanglement.optical_flow import BackgroundPreservationModule
from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor

# WAN2.2 imports
from src.pipelines.wan2_2_v2v_pipeline import Wan2_2MoEBlendedPipeline
from src.models.disentanglement.optical_flow import BackgroundPreservationModule
from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor

# WAN2.2 + LoRA + TIC-FT imports (NEW)
from src.models.adapters import (
    apply_lora_to_pipeline,
    TICFTBufferManager,
    RollingWindowInferenceEngine,
)
from src.pipelines.wan2_2_v2v_pipeline import Wan2_2MoEBlendedPipeline
```

---

## 🎯 When to Use Each

| Use Case | Recommend |
|----------|-----------|
| **Baseline comparison** | WAN2.1 |
| **Standard inference** | WAN2.2 (1.3B) |
| **Best quality (more VRAM)** | WAN2.2 (14B) |
| **Domain-specific tuning** | WAN2.2 + LoRA + TIC-FT |
| **Limited training data OK** | WAN2.2 + LoRA + TIC-FT |
| **Long video (81 frames)** | WAN2.2 + LoRA + TIC-FT |

---

## 📚 Documentation

| File | WAN2.1 | WAN2.2 | WAN2.2+LoRA |
|------|--------|--------|------------|
| `instruction.md` | ✅ | ✅ | ✅ |
| `AGENTS.md` | ✅ | ✅ | - |
| **`WAN_PIPELINE_COMPARISON.md`** | ✅ | ✅ | ✅ |
| **`WAN_CODE_STRUCTURE.md`** | ✅ | ✅ | ✅ |
| `TIC_FT_IMPLEMENTATION_GUIDE.md` | - | - | ✅ |
| `TIC_FT_QUICK_REFERENCE.md` | - | - | ✅ |

---

## ✅ Quick Checklist

```bash
# Test WAN2.1
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
./run_wan_inference.sh 0 "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  data/manifests/val.jsonl /tmp/test_wan 1

# Test WAN2.2
./run_wan2_2_inference.sh 0 "Wan-AI/Wan2.2-T2V-1.3B-Diffusers" \
  data/manifests/val.jsonl /tmp/test_wan2.2 1

# Test WAN2.2 + LoRA (quick train)
python3 scripts/train/train_wan2_2_lora_ticft.py \
  --manifest data/manifests/train.jsonl \
  --epochs 1 --max_samples 5 --output_dir /tmp/test_lora

# Test inference with LoRA
python3 scripts/infer/infer_wan2_2_lora_ticft.py \
  --manifest data/manifests/val.jsonl \
  --checkpoint /tmp/test_lora/lora_epoch1_step5.pt \
  --output_dir /tmp/test_lora_inf \
  --max_samples 1 --use_tic_ft_buffers
```

---

**Tách xong!** 🎉

- **WAN2.1**: Baseline, không training
- **WAN2.2**: Enhanced, inference only
- **WAN2.2+LoRA+TIC-FT**: Fine-tuning enabled, rolling window

Chi tiết xem: `WAN_PIPELINE_COMPARISON.md` + `WAN_CODE_STRUCTURE.md`
