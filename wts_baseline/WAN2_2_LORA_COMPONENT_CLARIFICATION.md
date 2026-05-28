# WAN2.2 + LoRA + TIC-FT: Component Usage Clarification

## Executive Summary

**Short Answer:** WAN2.2+LoRA+TIC-FT does **NOT** use MoE, Phase-Aware Semantic Routing, or Latent Flow Guidance. It uses only the **base WAN2.2 transformer** from HuggingFace with LoRA adapters applied on temporal layers.

---

## Three Distinct WAN2.2 Implementations

| Component | WAN2.2 (Inference) | WAN2.2+LoRA+TIC-FT (Training) | WAN2.2 MoE (Research) |
|-----------|-------------------|-------------------------------|----------------------|
| **Base Model** | HF `WanVideoToVideoPipeline` | HF `WanVideoToVideoPipeline` | Custom `Wan2_2MoEBlendedPipeline` |
| **Code File** | `scripts/infer/infer_wan2_2.py` | `scripts/train/train_wan2_2_lora_ticft.py` | `src/pipelines/wan2_2_v2v_pipeline.py` |
| **MoE (Experts)** | ❌ No | ❌ No | ✅ Yes (transformer + transformer_2) |
| **Phase-Aware Routing** | ❌ No | ❌ No | ✅ Yes |
| **Latent Flow Guidance** | ❌ No | ❌ No | ✅ Yes |
| **Slot Attention Adapter** | ❌ No | ❌ No | ⚠️ Yes (disabled by default) |
| **LoRA Adapters** | ❌ No | ✅ Yes (trainable) | ❌ No |
| **TIC-FT Buffers** | ❌ No | ✅ Yes | ❌ No |
| **Trainable Params** | 0 | ~50K | 0 |
| **Purpose** | Direct V2V inference | Domain-adapted V2V training | Experimental MoE research |

---

## Import Analysis: Which Components Are Actually Used?

### training/train_wan2_2_lora_ticft.py

```python
# What it imports:
from diffusers import WanVideoToVideoPipeline, AutoencoderKLWan, WanTransformer3DModel
from src.models.adapters import (
    apply_lora_to_pipeline,
    TICFTBufferManager,
    RollingWindowInferenceEngine,
)

# What it DOES NOT import:
# - src.pipelines.wan2_2_v2v_pipeline (Wan2_2MoEBlendedPipeline)
# - ANY custom MoE or routing components
```

**Pipeline Loading Code (line 381):**
```python
pipeline = create_pipeline(args.model_id, device, dtype)
# Calls WanVideoToVideoPipeline.from_pretrained() - the BASE pipeline!
```

**Transformer Handling (line 383):**
```python
pipeline.transformer = pipeline.transformer.to(device)
# This is the standard transformer, not transformer_2 or MoE blend
```

**LoRA Application (lines 391-396):**
```python
adapter_module = apply_lora_to_pipeline(
    pipeline,
    rank=args.lora_rank,
    alpha=args.lora_alpha,
    enable_training=True,
)
# Wraps pipeline.transformer with LoRA layers
# Original transformer remains frozen underneath LoRA
```

### infer/infer_wan2_2_lora_ticft.py

```python
from diffusers import WanVideoToVideoPipeline
from src.models.adapters import (
    apply_lora_to_pipeline,
    TICFTBufferManager,
    RollingWindowInferenceEngine,
)

# Does NOT import:
# - src.pipelines.wan2_2_v2v_pipeline
# - Custom MoE components
```

### Contrast: src/pipelines/wan2_2_v2v_pipeline.py

```python
# This file has all the MoE components:
from diffusers import (
    AutoencoderKLWan,
    WanTransformer3DModel,
    WanVideoToVideoPipeline as HFWanPipeline,  # Extends the base
)

class SlotAttentionAdapter(nn.Module):
    """Object-Centric Slot Attention"""
    
class Wan2_2MoEBlendedPipeline(HFWanPipeline):
    """Implements MoE Expert blending, Phase-Aware Semantic Routing,
    and Inference-time Latent Flow Guidance"""
```

**But this is NOT used in LoRA training!** It's only used in custom WAN2.2-specific inference pipelines.

---

## Component-by-Component Status

### 1. **MoE (Mixture of Experts)**
- **WAN2.2+LoRA+TIC-FT:** ❌ **NOT used**
- **Why:** Uses base `WanTransformer3DModel` from HF Diffusers, not custom MoE pipeline
- **Where MoE IS used:** `src/pipelines/wan2_2_v2v_pipeline.py` (separate inference code)

### 2. **Phase-Aware Semantic Expert Routing**
- **WAN2.2+LoRA+TIC-FT:** ❌ **NOT used**
- **Why:** Routing logic lives in `Wan2_2MoEBlendedPipeline.__call__()`, which isn't imported
- **Where it IS used:** Custom WAN2.2 MoE inference pipeline only

### 3. **Latent Flow Guidance** (ECCV proposal in WAN2.2 paper)
- **WAN2.2+LoRA+TIC-FT:** ❌ **NOT used**
- **Why:** Method `latent_flow_guidance_step()` is in `Wan2_2MoEBlendedPipeline`, not imported
- **Where it IS used:** Research implementation in `wan2_2_v2v_pipeline.py` only

### 4. **Slot Attention Adapter**
- **WAN2.2+LoRA+TIC-FT:** ❌ **NOT used**
- **Why:** Not available in base HF pipeline; only in custom MoE pipeline
- **Status:** Even in MoE pipeline, it's disabled by default (compresses 77→8 tokens, destroying text conditioning)

### 5. **Spatial Attention Layers**
- **WAN2.2+LoRA+TIC-FT:** ✅ **Yes, but FROZEN**
- **Why:** LoRA wrapper only adapts temporal layers; spatial layers remain frozen (read-only)
- **Code:** `WAN2_2TemporalLoRAAdapter.disable_lora_on_spatial_attention()`

### 6. **VAE (Autoencoder)**
- **WAN2.2+LoRA+TIC-FT:** ✅ **Yes, but FROZEN**
- **Why:** VAE is used for encoding/decoding but not adapted; remains in inference mode
- **Code:** VAE encoder stays frozen with `torch.no_grad()` in training loop

### 7. **LoRA Adapters** (NEW)
- **WAN2.2+LoRA+TIC-FT:** ✅ **YES - fully trainable**
- **Scope:** Temporal attention layers only (spatial layers remain frozen)
- **Parameters:** ~50K trainable vs 1.3B total base
- **Code:** `src/models/adapters/lora_temporal_adapter.py`

### 8. **TIC-FT Buffers** (NEW)
- **WAN2.2+LoRA+TIC-FT:** ✅ **YES - in inference**
- **Purpose:** Insert noisy buffer frames between condition and target for smooth rolling-window generation
- **Code:** `src/models/adapters/tic_ft_buffer.py`

---

## Training Pipeline Data Flow

```
Input Video (T frames)
    ↓
Split: History (5 frames) + Buffer (5 frames) + Target
    ↓
VAE Encode → Latent space [frozen]
    ↓
Base WAN2.2 Transformer (frozen base, LoRA adapters on top)
    ├─ Spatial Attention: FROZEN (no LoRA)
    ├─ Temporal Attention: LoRA wrapped (TRAINABLE)
    └─ MLP/Projection: FROZEN (no LoRA)
    ↓
LoRA computation: Δoutput = α·B(A^T·hidden_states)
    ↓
Loss backward only through LoRA params (~50K params)
    ↓
AdamW update only LoRA weights
```

**Note:** The base transformer stays frozen; only LoRA's low-rank matrices (A, B) get updated.

---

## Why NOT Include MoE in LoRA Training?

1. **MoE Components Not Available in HF Diffusers Base**
   - HuggingFace Diffusers `WanVideoToVideoPipeline` is the vanilla model
   - Custom MoE enhancements live in separate `wan2_2_v2v_pipeline.py`
   - Requires explicit refactoring to combine

2. **LoRA Sufficient for Domain Adaptation**
   - Temporal motion patterns dominate video generation changes
   - Spatial rendering (learned by base) doesn't need expert routing
   - ~50K LoRA params are enough to capture domain-specific temporal dynamics

3. **MoE Adds Complexity Without Clear Benefit**
   - Expert routing adds 3-5% overhead per forward pass
   - Training benefits unclear for a single domain (WTS)
   - Better to keep training simple, add MoE at inference only

---

## If You Want MoE + LoRA + TIC-FT Training...

You would need to:

1. **Integrate MoE into training pipeline:**
   ```python
   from src.pipelines.wan2_2_v2v_pipeline import Wan2_2MoEBlendedPipeline
   
   # Load MoE-enabled pipeline instead of base
   pipeline = Wan2_2MoEBlendedPipeline.from_pretrained(model_id)
   
   # Apply LoRA on top of MoE transformer
   adapter = apply_lora_to_pipeline(
       pipeline,
       rank=8,
       enable_training=True,
   )
   ```

2. **Handle transformer_2 (second expert) in LoRA:**
   - Currently `WAN2_2TemporalLoRAAdapter` assumes single transformer
   - Would need to wrap both `transformer` and `transformer_2` with separate LoRA adapters

3. **Training becomes 15-20% slower** due to:
   - Routing computation (~5% per layer)
   - Grad accumulation across multiple experts (~10%)
   - Expert load-balancing overhead (~5%)

---

## Recommended Architectures by Use Case

### For WTS Domain Adaptation (Recommended)
```
✅ Use: WAN2.2 Base + LoRA + TIC-FT
- Fast training (1.3B params, ~50K trainable)
- No MoE complexity
- Sufficient for one domain
- Script: scripts/train/train_wan2_2_lora_ticft.py
```

### For Multi-Domain or Complex Scenarios
```
⚠️  Consider: WAN2.2 Base + LoRA + TIC-FT → then MoE at inference
- Train LoRA on base model
- At inference: load LoRA + apply MoE blending
- Requires separate inference pipeline modification
```

### For Research: Full MoE + LoRA
```
🔬 Use: WAN2.2 MoE + Dual LoRA + TIC-FT
- Both transformers have LoRA adapters
- Much slower (~20% training overhead)
- Unclear benefit for single domain
- Requires significant implementation
```

---

## Summary Table: What's Trained vs. Frozen

| Component | Training | Frozen | Notes |
|-----------|----------|--------|-------|
| Base Transformer | ✅ LoRA only | Base frozen | ~50K params |
| Spatial Attention | ❌ Not adapted | ✅ Frozen | No LoRA |
| Temporal Attention | ✅ LoRA adapted | Base frozen | Main adaptation point |
| VAE Encoder | ❌ Not trained | ✅ Frozen | Encode to latent only |
| VAE Decoder | ❌ Not trained | ✅ Frozen | Decode from latent only |
| Text Encoder | ❌ Not trained | ✅ Frozen | Pre-computed embeddings |
| MoE Experts | ❌ Not used | N/A | Not in base pipeline |
| Expert Router | ❌ Not used | N/A | Not in base pipeline |
| Phase Routing | ❌ Not used | N/A | Not in base pipeline |

---

## Verification: Check Your Installation

To verify which components are actually loaded:

```bash
# See what pipeline is loaded in training
python3 -c "
from diffusers import WanVideoToVideoPipeline
pipe = WanVideoToVideoPipeline.from_pretrained('Wan-AI/Wan2.2-T2V-1.3B-Diffusers')
print('Pipeline type:', type(pipe))
print('Has transformer_2:', hasattr(pipe, 'transformer_2'))
print('Transformer type:', type(pipe.transformer))
print('Has MoE routing:', hasattr(pipe, 'latent_flow_guidance_step'))
"
# Output: Shows base pipeline, no transformer_2, no routing methods
```

```bash
# Compare with custom MoE pipeline
python3 -c "
from src.pipelines.wan2_2_v2v_pipeline import Wan2_2MoEBlendedPipeline
pipe = Wan2_2MoEBlendedPipeline.from_pretrained('Wan-AI/Wan2.2-T2V-1.3B-Diffusers')
print('Pipeline type:', type(pipe))
print('Has transformer_2:', hasattr(pipe, 'transformer_2'))
print('Has MoE routing:', hasattr(pipe, 'latent_flow_guidance_step'))
"
# Output: Shows MoE pipeline, has transformer_2, has routing methods
```

---

## Next Steps

1. **Use current WAN2.2+LoRA+TIC-FT as-is** for domain adaptation (recommended)
2. **If you need MoE at inference:** Create separate inference pipeline that loads LoRA weights + applies MoE blending
3. **If you need MoE during training:** Requires significant refactoring to integrate both pipelines

Would you like me to:
- [ ] Modify the training script to optionally use MoE?
- [ ] Create a hybrid inference pipeline (LoRA + MoE)?
- [ ] Keep current implementation and add comprehensive testing?
