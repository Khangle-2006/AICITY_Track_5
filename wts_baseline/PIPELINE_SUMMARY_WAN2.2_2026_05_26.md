# WTS Video Forecasting Pipeline - Wan2.2 Tóm Tắt (2026-05-26)

## 0. Tóm Tắt Baseline Wan2.2 - Experimental Version

### Status (2026-05-26) ⚠️ EXPERIMENTAL
| Aspect | Giá Trị |
|--------|--------|
| **Pipeline** | Wan2.2 MoE Text-to-Video (Wan-AI) |
| **Status** | ⚠️ Experimental - NOT RECOMMENDED for production |
| **Base Model** | Wan2.2 Flow-DiT (Mixture of Experts) |
| **Model ID** | `Wan-AI/Wan2.2-T2V-MoE-1.3B-Diffusers` (or 14B) |
| **Motivation** | Test higher-capacity MoE variant vs Wan2.1 |
| **Trade-off** | Lower metrics but potentially better scaling |

### Measured Performance (Validation - 4 Samples)

```
╔════════════════════════════════════════════════════════════════╗
║           MEASURED VALIDATION METRICS (WAN2.2)                 ║
╠════════════════════════════════════════════════════════════════╣
║ Metric        │ Wan2.2 │ Wan2.1 Baseline │ Δ (Worse)         ║
╠════════════════════════════════════════════════════════════════╣
║ PSNR ↑        │ 20.81  │ 26.89           │ -6.08 dB (-23%)   ║
║ SSIM ↑        │ 0.9002 │ 0.9470          │ -0.0468 (-5%)     ║
║ LPIPS ↓       │ 0.1192 │ 0.0723          │ +0.0469 (+65%)    ║
║ CLIP-S ↑      │ 0.2495 │ 0.2526          │ -0.0031 (-1.2%)   ║
║ FVD ↓         │ 517.33 │ 197.13          │ +320.2 (+162%)    ║
╠════════════════════════════════════════════════════════════════╣
║ VERDICT       │ ❌ Not Recommended │ Lower on all metrics       ║
╚════════════════════════════════════════════════════════════════╝
```

**Kết Luận**:
- ❌ **Wan2.2 performs worse than Wan2.1 on all 5 metrics**
- PSNR down 6 dB (-23%), FVD up 320 points (+162%)
- Not suitable for competition submission
- Kept for research/ablation studies only

---

## 1. Tổng Quan Wan2.2

### 1.1 Motivation for Wan2.2 Testing

| Khía Cạnh | Wan2.1 | Wan2.2 | Lý Do |
|----------|--------|--------|--------|
| **Architecture** | Single expert DiT | Mixture of Experts (MoE) | Scaling law: MoE → higher capacity |
| **Inference** | T2V (text→video) | V2V (video→video) | Closer to video prediction task |
| **Hidden Dim** | 1152 | Varies | MoE may have different scaling |
| **Parameters** | 1.3B | 1.3B (effective) | Similar param count |
| **Training Data** | 700M+ clips | Similar | Same pretrained scale |
| **Measured PSNR** | 26.89 dB | 20.81 dB | ❌ Wan2.2 underperforms |

### 1.2 Why Wan2.2 Underperforms

**Hypotheses**:
1. **MoE Routing Inefficiency**: Expert selection may not be optimal for short video chunks
2. **Sparse Activation**: Only subset of experts active → reduced capacity utilization
3. **V2V vs T2V**: Video-to-video mode may lose semantic information from text conditioning
4. **Fine-tuning Mismatch**: Pretrained on different dataset distribution than WTS
5. **Hyperparameter Misalignment**: CFG, guidance_scale not tuned for Wan2.2 specifics

### 1.3 Use Cases for Wan2.2

| Sử Dụng | Khuyến Nghị |
|---------|-----------|
| **Production Submission** | ❌ NO - Use Wan2.1 |
| **Ablation Study** | ✅ OK - Compare architectures |
| **Ensemble** | ⚠️ Maybe - Ensemble with Wan2.1 might help |
| **Scaling Research** | ✅ OK - Understand MoE limitations |
| **Quick Iteration** | ⚠️ Maybe - Faster inference if no quality loss acceptable |

---

## 2. Wan2.2 Architecture

### 2.1 Key Differences from Wan2.1

```
┌─ COMPARISON: WAN2.1 vs WAN2.2 ─────────────────────────────┐
│                                                             │
│ WAN2.1 (Baseline)                                           │
│ ├─ Dense Transformer DiT                                   │
│ ├─ All parameters active (no gating)                       │
│ ├─ Hidden size: 1152                                       │
│ ├─ Layers: 28-50                                           │
│ └─ Inference: T2V (text→video conditioning)                │
│                                                             │
│ WAN2.2 (Experimental MoE)                                   │
│ ├─ Mixture of Experts (MoE) DiT                            │
│ ├─ Only selected experts active per token                  │
│ ├─ Top-K routing (typically K=2 experts)                   │
│ ├─ Lower per-expert hidden dims, more experts              │
│ ├─ Sparse computation → memory efficient but possibly       │
│ │  inferior quality if routing is suboptimal               │
│ └─ Inference: V2V (video→video conditioning)               │
│    with learned gating network                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Mixture of Experts Layer

```python
class MoETransformerBlock(nn.Module):
    """
    Sparse expert layer with top-k routing.
    """
    
    def __init__(
        self,
        hidden_size=1152,
        num_experts=8,
        expert_hidden_size=288,  # 1152 / 4
        top_k=2,
        **kwargs
    ):
        super().__init__()
        
        # Gate network (router)
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Experts (sparse layers)
        self.experts = nn.ModuleList([
            TransformerFFN(hidden_size, expert_hidden_size, activation="gelu")
            for _ in range(num_experts)
        ])
        
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, x):
        """
        Args:
            x: [B*T, num_patches, hidden_size]
        
        Returns:
            output: [B*T, num_patches, hidden_size]
        """
        # Routing logits
        router_logits = self.gate(x)  # [B*T, num_patches, num_experts]
        
        # Select top-k experts
        router_weights, expert_indices = torch.topk(
            torch.softmax(router_logits, dim=-1),
            k=self.top_k, dim=-1
        )  # [B*T, num_patches, K], [B*T, num_patches, K]
        
        # Expert computation (sparse)
        outputs = []
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]  # [B*T, num_patches]
            expert_weight = router_weights[:, :, k]  # [B*T, num_patches]
            
            # Gather outputs from selected experts
            expert_output = torch.stack([
                self.experts[expert_idx[i]](x[i])
                for i in range(x.size(0))
            ])
            
            outputs.append(expert_weight.unsqueeze(-1) * expert_output)
        
        # Combine expert outputs
        output = sum(outputs)  # [B*T, num_patches, hidden_size]
        
        return output
```

### 2.3 V2V Pipeline (Video-to-Video Mode)

**Difference from T2V**:
```
T2V (Wan2.1):
  History frames + Text caption
         ↓
  VAE Encode history → latents
  UMT5 Encode caption → text_embedding
  DiT denoise with CFG
  VAE Decode → RGB
  
V2V (Wan2.2):
  History frames (no text!)
         ↓
  VAE Encode history → latents_history
  DiT denoise (uses warped history as context, not text)
  Apply learned gating based on video features
  VAE Decode → RGB
```

**Why V2V might fail on WTS**:
- Loss of semantic information (caption not used)
- Relies on implicit motion patterns from history
- May not capture phase-dependent behavior (pre-recognition vs action)

---

## 3. Wan2.2 Pipeline Configuration

### 3.1 Inference Parameters

```python
# Model & Resolution (same as Wan2.1)
model_id = "Wan-AI/Wan2.2-T2V-MoE-1.3B-Diffusers"
height = 480
width = 832

# Denoising (slightly different for MoE)
num_inference_steps = 50
guidance_scale = 5.0          # Start CFG
min_guidance = 1.5            # Higher than Wan2.1 (1.0)
strength = 0.7

# MoE-Specific
top_k = 2                      # Experts to activate per token
router_z_loss_coeff = 0.01    # Balance auxiliary loss (only during training)
router_aux_loss_coeff = 0.01

# Autoregressive
chunk_size = 81
overlap_frames = 2

# Optional: Lambda weighting (MoE specific)
lambda_weight = 0.1           # Trade-off parameter for expert selection
```

### 3.2 Shell Script Usage

```bash
#!/bin/bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Quick test (1 sample, ~45s)
python3 scripts/infer/infer_wan2_2.py \
  --manifest data/manifests/val_sota.jsonl \
  --output_dir /tmp/test_wan2_2 \
  --model_id Wan-AI/Wan2.2-T2V-MoE-1.3B-Diffusers \
  --height 480 --width 832 \
  --output_width 1280 --output_height 720 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --min_guidance 1.5 \
  --use_background_preservation \
  --use_alpha_compositing \
  --max_samples 1

# Or use shell script
./run_wan2_2_inference.sh 0 Wan-AI/Wan2.2-T2V-MoE-1.3B-Diffusers \
  data/manifests/val_sota.jsonl prediction_wan2_2_test 5
```

---

## 4. Performance Analysis

### 4.1 Detailed Metric Breakdown

| Metric | Wan2.2 | Wan2.1 | V2 | SOTA | Notes |
|--------|--------|--------|-----|------|-------|
| PSNR ↑ | 20.81 | **26.89** | 10.69 | 17.44 | Wan2.2 performs like SOTA, worse than Wan2.1 |
| SSIM ↑ | 0.9002 | **0.9470** | 0.2793 | 0.4710 | Still good, but 5% below Wan2.1 |
| LPIPS ↓ | 0.1192 | **0.0723** | 0.7449 | 0.8233 | 65% worse than Wan2.1 |
| CLIP-S ↑ | 0.2495 | 0.2526 | 20.91 | N/A | Essentially same (V2V doesn't use text) |
| FVD ↓ | 517.33 | **197.13** | N/A | 206.85 | 162% worse (temporal quality issues) |

### 4.2 Why FVD Degrades So Much

**FVD = Fréchet Video Distance** (measures temporal coherence)

**Hypothesis**:
1. **V2V mode** loses global semantic context (no text) → disjointed temporal evolution
2. **Expert routing** instability → some frames use different expert subsets → temporal discontinuity
3. **Higher min_guidance (1.5)** → forces more saturated predictions → temporal artifacts
4. **Chunk boundaries** → worse transitions due to expert switching

**Evidence**:
- PSNR only -6 dB (pixel-level is okay)
- FVD +320 (temporal structure broken)
- Suggests: per-frame quality okay, but temporal coherence poor

---

## 5. Comparison with Alternatives

### 5.1 When to Use Wan2.2 vs Wan2.1

| Scenario | Use Wan2.1 | Use Wan2.2 | Reason |
|----------|-----------|-----------|--------|
| **Production** | ✅ YES | ❌ NO | Wan2.1 has 26.89 vs 20.81 PSNR |
| **Ensemble** | ✅ Primary | ⚠️ Secondary | Combine predictions (if diversity helps) |
| **Quick Turnaround** | ❌ NO | ⚠️ Faster? | Wan2.2 not faster, just worse |
| **Scaling Study** | ⚠️ Maybe | ✅ YES | Understand MoE trade-offs |
| **Ablation** | ✅ Baseline | ✅ Test | Compare architecture impact |
| **Research** | ✅ Latest | ✅ Experimental | Both useful for analysis |

### 5.2 Ensemble Strategy (Optional)

If combining Wan2.1 + Wan2.2:

```python
def ensemble_predictions(wan2_1_frames, wan2_2_frames, alpha=0.7):
    """
    Blend two predictions.
    
    Args:
        wan2_1_frames: [T, H, W, 3] uint8 (quality 26.89 PSNR)
        wan2_2_frames: [T, H, W, 3] uint8 (quality 20.81 PSNR)
        alpha: weight for Wan2.1 (default 0.7 = 70% Wan2.1, 30% Wan2.2)
    
    Returns:
        ensemble: [T, H, W, 3] uint8
    """
    wan2_1_float = wan2_1_frames.astype(np.float32)
    wan2_2_float = wan2_2_frames.astype(np.float32)
    
    ensemble = alpha * wan2_1_float + (1 - alpha) * wan2_2_float
    ensemble = np.clip(ensemble, 0, 255).astype(np.uint8)
    
    return ensemble

# Expected improvement: marginal (1-2% gain at best)
# Cost: 2× inference time
# NOT RECOMMENDED unless time permits
```

---

## 6. Troubleshooting Wan2.2

### 6.1 Common Issues

#### Issue 1: Poor Temporal Coherence (High FVD)

**Symptoms**: Generated frames flicker, jump between chunks

**Causes**: V2V mode without text context

**Solutions**:
```bash
# 1. Increase overlap for smoother transitions
--overlap_frames 4  # from 2

# 2. Reduce expert switching (if available)
--top_k 1  # Force single expert (not recommended)

# 3. Use stronger CFG to maintain coherence
--guidance_scale 6.0  # from 5.0

# 4. Disable chunking if possible (won't help much)
# Wan2.2 structural issue cannot be overcome

# BEST SOLUTION: Use Wan2.1 instead
```

#### Issue 2: Low CLIP-S (Semantic Mismatch)

**Symptoms**: Generated content doesn't match captions

**Root Cause**: V2V mode doesn't use text conditioning!

**Solution**: 
- Expected for V2V. Not fixable without switching to Wan2.1 T2V mode.

---

## 7. Command Reference (Wan2.2)

### 7.1 Setup
```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 7.2 Quick Test (1 sample)
```bash
python3 scripts/infer/infer_wan2_2.py \
  --manifest data/manifests/val_sota.jsonl \
  --output_dir /tmp/test_wan2_2 \
  --model_id Wan-AI/Wan2.2-T2V-MoE-1.3B-Diffusers \
  --max_samples 1
```

### 7.3 Full Validation (4-5 samples)
```bash
python3 scripts/infer/infer_wan2_2.py \
  --manifest data/manifests/val_sota.jsonl \
  --output_dir prediction_wan2_2_val \
  --model_id Wan-AI/Wan2.2-T2V-MoE-1.3B-Diffusers \
  --use_background_preservation \
  --use_alpha_compositing \
  --use_dynamic_cfg \
  --max_samples 5
```

### 7.4 Evaluate
```bash
python3 scripts/eval/eval_sota.py \
  --output_dir prediction_wan2_2_val \
  --manifest data/manifests/val_sota.jsonl \
  --device cuda
```

---

## 8. Conclusion & Recommendation

### ❌ NOT RECOMMENDED FOR PRODUCTION

**Wan2.2 Experimental Pipeline**:
- ✅ Interesting MoE architecture
- ✅ Potential for future scaling
- ❌ **Lower metrics on all fronts** (PSNR -23%, FVD +162%)
- ❌ V2V mode loses text context
- ❌ Temporal coherence issues

### ✅ USE WAN2.1 INSTEAD

| Metric | Wan2.1 | Wan2.2 | Winner |
|--------|--------|--------|--------|
| PSNR | 26.89 | 20.81 | ✅ Wan2.1 |
| SSIM | 0.9470 | 0.9002 | ✅ Wan2.1 |
| LPIPS | 0.0723 | 0.1192 | ✅ Wan2.1 |
| CLIP-S | 0.2526 | 0.2495 | ≈ Tie |
| FVD | 197.13 | 517.33 | ✅ Wan2.1 |

---

## 9. File Locations

| Item | Path |
|------|------|
| **Wan2.2 Inference** | `scripts/infer/infer_wan2_2.py` |
| **Wan2.2 Pipeline** | `src/pipelines/wan2_2_v2v_pipeline.py` |
| **Shell Script** | `run_wan2_2_inference.sh` |
| **Evaluator** | `scripts/eval/eval_sota.py` |
| **Outputs (Val)** | `prediction_wan2.2_val_5/` |
| **Metrics** | `prediction_wan2.2_val_5/eval_results_sota.json` |

---

## 10. Next Steps

### If Using Wan2.2:
1. ⚠️ Understand it's experimental and lower quality than Wan2.1
2. Run quick validation to confirm metrics match
3. Consider ensemble with Wan2.1 (if time permits)
4. Document findings for research/ablation

### Recommended Path:
1. ✅ Use **Wan2.1** for official submission
2. ✅ Keep Wan2.2 results for comparison/analysis
3. ⏳ Maybe revisit Wan2.2 for future iterations

---

**Last Updated**: 2026-05-26  
**Status**: ⚠️ Experimental (NOT RECOMMENDED)  
**Primary Baseline**: Wan2.1 SOTA  
**Wan2.2 Measured Performance**: PSNR 20.81 dB | SSIM 0.9002 | LPIPS 0.1192 | FVD 517.33  
**Maintainer**: WTS Baseline Team
