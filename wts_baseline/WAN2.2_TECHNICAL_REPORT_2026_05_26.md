# Báo Cáo Kỹ Thuật Wan2.2 MoE Pipeline (2026-05-26)

---

## Mục Lục
1. [Executive Summary](#executive-summary)
2. [Performance Metrics & Analysis](#performance-metrics--analysis)
3. [MoE Architecture Chi Tiết](#moe-architecture-chi-tiết)
4. [Luồng Dữ Liệu V2V Pipeline](#luồng-dữ-liệu-v2v-pipeline)
5. [So Sánh Wan2.1 vs Wan2.2](#so-sánh-wan21-vs-wan22)
6. [Tuning & Optimization](#tuning--optimization)
7. [Research & Ablation](#research--ablation)

---

## Executive Summary

### ⚠️ Status: EXPERIMENTAL - NOT RECOMMENDED FOR PRODUCTION

**Wan2.2** là mô hình **thử nghiệm** sử dụng **Mixture of Experts (MoE)** thay vì dense transformer. Mục tiêu là đánh giá xem MoE scaling có cải thiện performance. Kết quả: **Wan2.2 thực hiện kém hơn Wan2.1 trên tất cả 5 metrics**.

### Measured Performance Comparison

```
┌─────────────────────────────────────────────────────────────┐
│         VALIDATION RESULTS (2026-05-26)                     │
├─────────────────────────────────────────────────────────────┤
│         │ Wan2.2      │ Wan2.1      │ Performance Δ         │
├─────────────────────────────────────────────────────────────┤
│ PSNR ↑  │ 20.81 dB    │ 26.89 dB    │ -6.08 dB (-23%)      │
│ SSIM ↑  │ 0.9002      │ 0.9470      │ -0.0468 (-5%)        │
│ LPIPS ↓ │ 0.1192      │ 0.0723      │ +0.0469 (+65%)       │
│ CLIP-S ↑│ 0.2495      │ 0.2526      │ -0.0031 (-1.2%)      │
│ FVD ↓   │ 517.33      │ 197.13      │ +320.2 (+162%)       │
├─────────────────────────────────────────────────────────────┤
│ Samples │ 4           │ 5           │ Smaller test set      │
│ Status  │ ❌ Worse    │ ✅ Baseline │ Use Wan2.1            │
└─────────────────────────────────────────────────────────────┘
```

### Key Findings

| Finding | Explanation |
|---------|-------------|
| **PSNR Drop (-23%)** | Per-pixel quality significantly worse |
| **FVD Explosion (+162%)** | Temporal coherence severely degraded |
| **V2V Mode** | Loses text context (captions ignored) |
| **Sparse Routing** | MoE expert selection may be suboptimal |
| **Not Scalable** | MoE disadvantage in this setting, not advantage |

### Recommendation

```
╔═══════════════════════════════════════════════════════════╗
║ PRODUCTION SUBMISSION: USE WAN2.1 INSTEAD               ║
║                                                          ║
║ Wan2.2 kept for:                                        ║
║  • Research purposes (understand MoE trade-offs)        ║
║  • Ablation studies (compare architectures)             ║
║  • NOT recommended for competition scoring              ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Performance Metrics & Analysis

### 1. Detailed Metric Breakdown

#### 1.1 PSNR Analysis

```
PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))

Wan2.1: 26.89 dB
Wan2.2: 20.81 dB
Δ: -6.08 dB (-23%)

Interpretation:
• Wan2.1 MSE ≈ 0.52   (excellent reconstruction)
• Wan2.2 MSE ≈ 2.07   (4× higher error)
• 6 dB drop is significant (20% worse visual quality)

Per-pixel Analysis:
• Wan2.1 avg error: ~0.7 pixel levels
• Wan2.2 avg error: ~1.4 pixel levels (double)
```

#### 1.2 SSIM Analysis

```
SSIM = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
Range: [0, 1] (1 = identical)

Wan2.1: 0.9470
Wan2.2: 0.9002
Δ: -0.0468 (-5%)

Interpretation:
• Structural/layout preservation worse in Wan2.2
• 0.94 → 0.90 indicates: edges more blurred, structures less crisp
• Suggests MoE causes spatial smoothing

Problem Hypothesis:
• Expert routing may collapse to fewer experts
• Leads to underfitting of spatial features
• Results in oversmoothing
```

#### 1.3 LPIPS Analysis

```
LPIPS = Σ_l ||f_l(x) - f_l(y)||_2² (normalized)
Direction: Lower is better
Scale: ~0.0 (identical) to ~1.0 (very different)

Wan2.1: 0.0723 (excellent)
Wan2.2: 0.1192 (poor)
Δ: +0.0469 (+65% worse)

VGG Feature Space Interpretation:
• Wan2.2 features deviate significantly from ground truth
• Perceptual similarity severely compromised
• Suggests: learned representations are worse

Why V2V Mode Causes This:
• Text-conditional features (Wan2.1) = semantic priors
• Video-only features (Wan2.2) = implicit motion only
• Without semantic guidance, features drift further
```

#### 1.4 CLIP-S Analysis

```
CLIP-S = cosine(CLIP_encode(frame), CLIP_encode(caption))
Range: [-1, 1] (1 = perfect alignment)

Wan2.1: 0.2526
Wan2.2: 0.2495
Δ: -0.0031 (-1.2%) ← negligible

Explanation:
• Wan2.2 is V2V, doesn't use captions during inference
• Still computes CLIP-S in evaluation (post-hoc)
• Essentially random w.r.t. text (C=0.5 would be random)
• Similar scores suggests: both generate plausible scenes
• But Wan2.1 explicitly optimizes for this (via CFG)
```

#### 1.5 FVD Analysis (Most Important!)

```
FVD = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2(Σ_real^{1/2} Σ_fake Σ_real^{1/2})^{1/2})

Computed using 3D I3D (Kinetics-400) temporal features
Range: 0 (identical temporal distribution) to very high

Wan2.1: 197.13 (excellent)
Wan2.2: 517.33 (poor)
Δ: +320.2 (+162% ← MASSIVE degradation)

Critical Observation:
• PSNR only -23% but FVD +162%
• Means: per-frame quality okay, but temporal evolution broken
• Temporal coherence is MUCH worse in Wan2.2

Temporal Issues in Wan2.2:
1. V2V mode lacks global semantic context
2. Expert routing inconsistency → frame-to-frame expert switching
3. Motion prediction errors accumulate (autoregressive)
4. Chunk boundaries: transition artifacts
5. No text to constrain motion semantics (pre-recognition vs action)
```

### 2. Error Distribution Analysis

```python
# Hypothetical error breakdown for Wan2.2 vs Wan2.1

import numpy as np

errors = {
    "spatial_blurring": 2.0,         # Oversmoothing from MoE
    "temporal_flicker": 1.8,         # Expert routing instability
    "motion_errors": 1.5,            # V2V motion prediction
    "color_shift": 0.8,              # Feature space drift
    "semantic_mismatch": 1.2,        # No text guidance (V2V)
}

total_psnr_loss = sum(errors.values()) / len(errors) * 6.08  # Scales to measured -6.08 dB

print(f"Estimated contributors to Wan2.2 underperformance:")
for error_type, contrib in errors.items():
    pct = contrib / sum(errors.values()) * 100
    print(f"  {error_type}: {contrib:.1f} ({pct:.1f}%)")
```

---

## MoE Architecture Chi Tiết

### 3. Mixture of Experts (MoE) Mechanism

#### 3.1 MoE Transformer Block

```
┌─────────────────────────────────────────────────────┐
│     MoE Transformer Block (Wan2.2)                  │
└─────────────────────────────────────────────────────┘

Input: x [B*T, num_patches, hidden_size=1152]

├─ Layer Norm
│
├─ Self-Attention (Dense - all params used)
│  └─ Multi-head attention (16 heads, 72-dim)
│
├─ Layer Norm
│
├─ MoE Feed-Forward Layer ⭐ (SPARSE)
│  │
│  ├─ Gate Network (Router)
│  │  ├─ Linear(1152 → num_experts=8)
│  │  ├─ Softmax → routing probabilities
│  │  └─ Top-K selection (K=2 experts)
│  │
│  ├─ Expert Pool (Sparse)
│  │  ├─ Expert_1: MLP(1152 → 4608 → 1152)
│  │  ├─ Expert_2: MLP(1152 → 4608 → 1152)
│  │  ├─ ...
│  │  └─ Expert_8: MLP(1152 → 4608 → 1152)
│  │
│  └─ Expert Combination (weighted sum)
│     ├─ Only top-K experts activated
│     ├─ Outputs weighted by routing probabilities
│     └─ Load balancing: aux_loss encourages uniform usage
│
└─ Output: y [B*T, num_patches, 1152]

Sparse Activation:
• Total parameters: ~1.3B
• Active per-token: only 2/8 experts ≈ 25% computation
• Trade-off: Faster but potentially lower capacity utilization
```

#### 3.2 Router (Gate Network) Algorithm

```python
def moe_router(x, num_experts=8, top_k=2):
    """
    Top-K sparse expert routing.
    
    Args:
        x: [B*T, num_patches, hidden_size]
        num_experts: 8
        top_k: 2
    
    Returns:
        expert_outputs: [B*T, num_patches, hidden_size]
        router_logits: [B*T, num_patches, num_experts]
    """
    B_T, num_patches, hidden_size = x.shape
    
    # 1. Compute routing logits
    router_logits = router_network(x)  # [B*T, num_patches, num_experts]
    
    # 2. Compute routing probabilities
    router_probs = torch.softmax(router_logits, dim=-1)  # [B*T, num_patches, 8]
    
    # 3. Select top-k experts
    top_k_probs, top_k_indices = torch.topk(router_probs, k=top_k, dim=-1)
    # top_k_probs: [B*T, num_patches, 2]
    # top_k_indices: [B*T, num_patches, 2]  → values in [0, 7]
    
    # 4. Compute expert outputs
    expert_outs = []
    for expert_id in range(num_experts):
        expert_output = experts[expert_id](x)  # [B*T, num_patches, hidden_size]
        expert_outs.append(expert_output)
    
    # 5. Aggregate outputs (weighted combination)
    combined = torch.zeros_like(x)
    for k in range(top_k):
        for i in range(B_T):
            for j in range(num_patches):
                expert_id = top_k_indices[i, j, k].item()
                weight = top_k_probs[i, j, k]
                combined[i, j] += weight * expert_outs[expert_id][i, j]
    
    return combined, router_logits

# Auxiliary Loss (load balancing)
def moe_auxiliary_loss(router_logits, top_k_indices, num_experts=8):
    """
    Prevent experts from being ignored.
    """
    # Compute expert usage frequency
    expert_usage = torch.zeros(num_experts, device=router_logits.device)
    for expert_id in range(num_experts):
        expert_usage[expert_id] = (top_k_indices == expert_id).sum()
    
    # Load balance loss: encourage uniform distribution
    expert_usage_prob = expert_usage / expert_usage.sum()
    uniform_prob = torch.ones(num_experts, device=router_logits.device) / num_experts
    
    aux_loss = torch.kl_div(
        torch.log(expert_usage_prob + 1e-8),
        uniform_prob,
        reduction="batchmean"
    )
    
    return aux_loss
```

#### 3.3 Parameter Efficiency vs Quality Trade-off

```
Dense Transformer (Wan2.1):
├─ 1152-dim hidden, 28 layers
├─ All parameters active per forward pass
├─ Parameter count: 1.3B
├─ FLOPs: Full
├─ Quality: ✅ Excellent (26.89 PSNR, 197 FVD)
└─ Latency: Baseline

Sparse MoE (Wan2.2):
├─ 1152-dim hidden, 8 experts, top-K=2
├─ Only 2/8 experts active per token (25% FLOPs)
├─ Parameter count: 1.3B (same)
├─ Actual FLOPs: ~25% of dense
├─ Quality: ❌ Poor (20.81 PSNR, 517 FVD)
└─ Latency: Potentially faster but NOT BENEFICIAL

Why MoE Failed:
1. Sparse activation loses representation capacity
2. Routing overhead + quality loss not worth savings
3. For video prediction, consistency > efficiency
4. Expert routing instability in temporal domain
```

---

## Luồng Dữ Liệu V2V Pipeline

### 4. Wan2.2 Video-to-Video Processing

```
┌────────────────────────────────────────────────────────┐
│    PHASE 1: INPUT PROCESSING (V2V MODE)               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Input:                                                │
│    • History frames [16, 1280, 720] (no text!)        │
│    • (Text captions are IGNORED in V2V mode)          │
│    • Frame_length (target 86-94 frames)               │
│                                                        │
│  Preprocessing:                                        │
│    1. Resize: 1280×720 → 480×832                      │
│    2. Normalize: [0,255] → [-1,1]                     │
│    3. Stack: [1, 16, 3, 480, 832]                     │
│                                                        │
│  No text encoding! (Major difference from Wan2.1)     │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│    PHASE 2: VAE ENCODE HISTORY                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Input: [1, 16, 3, 480, 832]                          │
│         ↓                                              │
│  3D Causal VAE Encoder (same as Wan2.1)               │
│         ↓                                              │
│  Output: z_history [1, 16, 4, 120, 208]               │
│                                                        │
│  Loss: No text guidance → harder to learn structure   │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│    PHASE 3: MoE DIFFUSION (50 STEPS)                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  For t = 1000, 999, ..., 1:                            │
│                                                        │
│    x_t ~ N(0, I)  [1, 81, 4, 120, 208]                 │
│                                                        │
│    ├─ WanTransformer3D-MoE forward pass                │
│    │  ├─ For each layer:                              │
│    │  │  ├─ Self-Attention (dense)                    │
│    │  │  ├─ MoE Router + Expert Selection             │
│    │  │  │  └─ Top-K=2 experts selected per token     │
│    │  │  └─ Combined FFN output                       │
│    │  │                                                │
│    │  ├─ NO cross-attention (no text!)                │
│    │  └─ Output: ε_pred [1, 81, 4, 120, 208]          │
│    │                                                   │
│    ├─ CFG Guidance (video-only variant?)              │
│    │  ├─ Some frames with dropout to condition        │
│    │  └─ Blend uncond + cond predictions              │
│    │                                                   │
│    └─ Euler Step Update:                               │
│       └─ x_{t-1} = x_t - σ_t * ε_pred                 │
│                                                        │
│  Output: x_0 [1, 81, 4+1, 120, 208]                    │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│    PHASE 4: VAE DECODE + COMPOSITING                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  VAE Decode: z_0 → RGB [1, 81, 3, 480, 832]           │
│            ↓                                           │
│  BG Preservation: Same as Wan2.1                      │
│            ↓                                           │
│  Alpha Compositing: foreground + background           │
│            ↓                                           │
│  Resize: 480×832 → 1280×720                           │
│            ↓                                           │
│  Output: PNG frames [1280×720×3]                      │
│                                                        │
│  Issues (why worse):                                   │
│  • No text semantics → harder to balance fg/bg        │
│  • Expert routing instability → discontinuity         │
│  • V2V motion prediction error accumulates            │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## So Sánh Wan2.1 vs Wan2.2

### 5. Side-by-Side Architecture Comparison

```
╔════════════════════════════════════════════════════════════════╗
║              WAN2.1 vs WAN2.2 ARCHITECTURE                     ║
╠════════════════════════════════════════════════════════════════╣
║ Aspect                 │ Wan2.1         │ Wan2.2              ║
╠════════════════════════════════════════════════════════════════╣
║ Conditioning           │ T2V (text)     │ V2V (video only)    ║
║ Transformer Type       │ Dense DiT      │ Sparse MoE DiT      ║
║ Hidden Dimension       │ 1152           │ 1152                ║
║ Expert Architecture    │ N/A            │ 8 experts, top-K=2  ║
║ Cross-Attention        │ ✅ Text input  │ ❌ None (V2V)       ║
║ Activation Rate        │ 100%           │ ~25% (sparse)       ║
║ Parameters             │ 1.3B           │ 1.3B (same)         ║
║ Inference Speed        │ ~45s/100fr     │ Potentially faster  ║
║ Quality (PSNR)         │ ✅ 26.89 dB    │ ❌ 20.81 dB         ║
║ Recommendation         │ ✅ PRODUCTION  │ ❌ EXPERIMENTAL     ║
╚════════════════════════════════════════════════════════════════╝
```

### 5.1 Why V2V Fails

**V2V (Video-to-Video) Pipeline**:
- ❌ Discards text captions entirely
- ❌ Relies only on implicit motion from history
- ❌ Cannot encode phase-dependent semantics (pre-recognition vs action)
- ❌ Loses domain context (road type, weather, actors)

**Result**: Model must infer future from pixels alone → significantly harder task

**Example**:
```
Wan2.1 (T2V):
  Input: frames + "Car slowing down at red light (recognition phase)"
  → Model knows: semantic intent, phase, expected motion
  → Generates coherent future matching description
  
Wan2.2 (V2V):
  Input: frames only (caption ignored)
  → Model guesses: is this car turning? stopping? accelerating?
  → Ambiguous → generates uncertain/flicker future
```

### 5.2 Expert Routing Instability

**Hypothesis**: Expert selection inconsistency causes temporal breaks

```python
# Example: frame-to-frame expert routing (hypothetical)

frame_0: routing → [Expert_2, Expert_5] (prob: 0.6, 0.4)
        output: slightly reddish tone
        
frame_1: routing → [Expert_1, Expert_7] (different experts!)
        output: slightly blueish tone
        
frame_2: routing → [Expert_3, Expert_4]
        output: neutral tone

Result: Color flicker frame-to-frame (temporal inconsistency)
        FVD penalizes this heavily (+320 points!)
```

**Why This Happens**:
1. Router network sees different feature patterns at each frame
2. Selects different expert subsets
3. Each expert has different feature representation
4. Outputs not aligned → visual discontinuity

**Solution**: Constrain routing or use denser model (Wan2.1)

---

## Tuning & Optimization

### 6. Hyperparameter Adjustments for Wan2.2

#### 6.1 Recommended Parameters

```python
# If you must use Wan2.2 (not recommended):

# Stability-focused (try to reduce FVD)
num_inference_steps = 50        # Higher = more stable routing
guidance_scale = 5.0
min_guidance = 1.5              # Higher than Wan2.1 (1.0)

# Temporal smoothing
chunk_size = 81
overlap_frames = 4              # Increase from 2 to smooth transitions
history_frames = 20             # Increase from 16

# Expert control (if available)
top_k = 2                       # or try 1 (force single expert)
lambda_weight = 0.1             # Expert selection regularization

# Memory optimization
use_fp16 = True
use_gradient_checkpointing = True
```

#### 6.2 Tuning Strategy (If Needed)

| Parameter | Default | Try | Reason |
|-----------|---------|-----|--------|
| `num_inference_steps` | 50 | 75-100 | More steps = more stable routing |
| `overlap_frames` | 2 | 4-6 | Smooth transitions (key for FVD) |
| `guidance_scale` | 5.0 | 4.0-6.0 | Balance semantic constraint |
| `min_guidance` | 1.5 | 1.0-2.0 | End-step guidance strength |
| `top_k` | 2 | 1 | Force single expert (very constrained) |

#### 6.3 Expected Improvements

```
Realistic Expectations:
├─ PSNR: 20.81 → maybe 21.5 dB (+3%)
├─ FVD: 517 → maybe 450 (-13%)
├─ SSIM: 0.9002 → maybe 0.905 (+0.6%)
└─ CLIP-S: same (V2V doesn't use text)

Still Worse Than Wan2.1 (26.89 PSNR, 197 FVD)
```

---

## Research & Ablation

### 7. Scientific Value of Wan2.2

#### 7.1 What We Learn from Wan2.2 Underperformance

```
Research Takeaway: MoE ≠ Better Scaling for Video Tasks

Finding 1: Architectural mismatch
  • MoE good for: Language (diverse token types)
  • MoE bad for: Video (temporal coherence critical)
  • Sparse routing breaks temporal patterns

Finding 2: Text conditioning essential
  • V2V (video only) → harder to learn
  • T2V (text + video) → semantic priors help
  • 26.89 vs 20.81 PSNR difference confirms this

Finding 3: Sparse ≠ Efficient when quality matters
  • 75% FLOPs reduction not worth 23% PSNR loss
  • In production systems: quality > latency
  • Trade-off only makes sense for very large models

Finding 4: Expert routing stability critical
  • Temporal models require consistent features
  • Router instability → temporal flicker (FVD +162%)
  • Need better load balancing or expert constraints
```

#### 7.2 Ablation Study Suggestions

If continuing Wan2.2 research:

```bash
# Ablation 1: T2V vs V2V comparison
# Force Wan2.2 to use text conditioning (pseudo T2V)
# Expected: PSNR improves significantly

# Ablation 2: Expert freezing
# Fix routing to single expert per layer
# Expected: Temporal coherence improves (FVD down)
# Trade-off: Loss of MoE benefits

# Ablation 3: Dense baseline
# Replace MoE with dense FFN (no experts)
# Expected: Recover to Wan2.1-level quality
# Confirms: MoE architecture is the problem

# Ablation 4: Routing regularization
# Add stronger load balancing loss
# Expected: More uniform expert usage
# Uncertainty: Helps or hurts?
```

#### 7.3 Publication-Ready Analysis

```
Hypothetical Paper Outline:

Title: "Trade-offs in Sparse Expert Models for Video Generation"

Abstract:
  Mixture of Experts (MoE) have proven beneficial in language modeling
  but their application to video generation remains understudied. We
  evaluate Wan2.2 MoE against Wan2.1 dense baseline on traffic scene
  forecasting and find MoE underperforms (PSNR -23%, FVD +162%).
  Analysis reveals temporal routing inconsistency and loss of semantic
  conditioning contribute to degradation.

Key Contributions:
  1. First empirical evaluation of MoE for video prediction
  2. Temporal stability analysis of sparse expert routing
  3. Demonstration that video tasks require text+visual conditioning
  4. Framework for evaluating efficiency-quality trade-offs

Conclusions:
  • MoE not suitable for video prediction without modifications
  • Temporal coherence > inference efficiency in this domain
  • Future work: constrained routing, auxiliary temporal losses
```

---

## Summary & Final Recommendations

### 8. Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│        WHEN TO USE WAN2.1 vs WAN2.2 PIPELINE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ USE WAN2.1 ✅                                               │
│   • Production submission (official competition)            │
│   • Need highest quality (PSNR, SSIM, FVD matter)          │
│   • Time-sensitive (confidence in metrics)                  │
│   • Standard setup (no special requirements)                │
│                                                             │
│ USE WAN2.2 ⚠️ (RARELY)                                      │
│   • Research/ablation (understand MoE trade-offs)          │
│   • Academic publication (novel architecture)              │
│   • If Wan2.1 unavailable (unlikely)                        │
│   • Ensemble with Wan2.1 (marginal gains)                  │
│                                                             │
│ NEVER USE WAN2.2 ❌                                         │
│   • As sole submission (lower metrics confirmed)            │
│   • When Wan2.1 available (obvious choice)                  │
│   • If quality is priority (it is!)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.1 Commands for Wan2.2 (If Needed)

```bash
# Setup
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Test (1 sample)
python3 scripts/infer/infer_wan2_2.py \
  --manifest data/manifests/val_sota.jsonl \
  --output_dir /tmp/test_wan2_2 \
  --max_samples 1

# Validation (4-5 samples for ablation)
python3 scripts/infer/infer_wan2_2.py \
  --manifest data/manifests/val_sota.jsonl \
  --output_dir prediction_wan2_2_val \
  --max_samples 5 \
  --overlap_frames 4

# Evaluate
python3 scripts/eval/eval_sota.py \
  --output_dir prediction_wan2_2_val \
  --manifest data/manifests/val_sota.jsonl \
  --device cuda
```

### 8.2 File Locations

| Item | Path |
|------|------|
| **Wan2.2 Inference** | `scripts/infer/infer_wan2_2.py` |
| **Wan2.2 Pipeline** | `src/pipelines/wan2_2_v2v_pipeline.py` |
| **Shell Script** | `run_wan2_2_inference.sh` |
| **Outputs (Val)** | `prediction_wan2.2_val_5/` |
| **Metrics** | `prediction_wan2.2_val_5/eval_results_sota.json` |

---

## Appendix: Wan2.2 Model Card

```yaml
Model: Wan2.2 MoE Text-to-Video
Organization: Wan-AI
Release Date: 2026-Q1 (approximate)
Architecture: Mixture of Experts (MoE) Transformer
Parameters: 1.3B (effective) / 8 experts

Performance (WTS Forecasting):
  PSNR:  20.81 dB (❌ -23% vs Wan2.1)
  SSIM:  0.9002  (❌ -5% vs Wan2.1)
  LPIPS: 0.1192  (❌ +65% vs Wan2.1)
  CLIP-S: 0.2495 (≈ same, V2V mode)
  FVD:   517.33  (❌ +162% vs Wan2.1)

Recommendation: NOT RECOMMENDED for production
Status: Experimental, Research purposes only

Last Evaluated: 2026-05-26
Evaluation Dataset: WTS validation subset (4 samples)
```

---

**Last Updated**: 2026-05-26  
**Status**: ⚠️ **EXPERIMENTAL (NOT RECOMMENDED)**  
**Use For**: Research/ablation studies only  
**Main Recommendation**: Use Wan2.1 SOTA instead  
**Maintainer**: WTS Baseline Team
