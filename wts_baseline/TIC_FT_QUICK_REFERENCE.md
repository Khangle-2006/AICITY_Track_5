"""
Quick Reference: TIC-FT Adapter Integration for WAN2.2

This guide shows how to integrate TIC-FT adapters into existing code.
Paper: https://arxiv.org/pdf/2506.00996
"""

# ============================================================================
# 1. QUICK START - Apply LoRA + TIC-FT to inference
# ============================================================================

import torch
from diffusers import WanVideoToVideoPipeline
from src.models.adapters import (
    apply_lora_to_pipeline,
    TICFTBufferManager,
)

# Load pipeline
pipeline = WanVideoToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-1.3B-Diffusers",
    torch_dtype=torch.bfloat16,
)

# Apply LoRA (auto-detects temporal attention layers)
adapter = apply_lora_to_pipeline(
    pipeline,
    rank=8,
    alpha=1.0,
    enable_training=False  # Inference mode
)

# Load pre-trained LoRA weights
adapter.load_lora_weights("checkpoints_lora/lora_epoch10.pt", device="cuda")

# Setup TIC-FT buffer manager
tic_ft = TICFTBufferManager(
    buffer_frames=5,          # Insert 5 noisy frames between chunks
    chunk_size=16,            # Generate 16 frames per chunk
    overlap_frames=4,         # Overlap 4 frames between chunks
    noise_schedule="cosine"   # Smooth noise ramp
)

# Use for generation
output = pipeline(
    video=history_frames,      # [B, T_hist, C, H, W]
    prompt_embeds=prompts,     # [B, L, D]
    guidance_scale=7.5,
    num_inference_steps=50,
)

# ============================================================================
# 2. TRAINING SETUP
# ============================================================================

from torch.optim import AdamW

# Load pipeline
pipeline = WanVideoToVideoPipeline.from_pretrained(model_id)

# Apply LoRA with training enabled
adapter = apply_lora_to_pipeline(
    pipeline,
    rank=8,
    enable_training=True  # Enable LoRA for training
)

# Only LoRA parameters are trainable
trainable_params = [p for p in adapter.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=5e-5)

print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
# Expected: ~50K (vs. 14B total)

# Training loop...
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward, backward, optimize
        loss.backward()
        optimizer.step()
    
    # Save only LoRA weights (frozen base not needed)
    adapter.save_lora_weights(f"lora_epoch{epoch}.pt")

# ============================================================================
# 3. ARCHITECTURE DETAILS
# ============================================================================

"""
LoRA Decomposition:
    Given frozen linear layer: y = W @ x
    LoRA update: y_new = W @ x + α * B @ A^T @ x
    Where:
    - A ∈ ℝ^(d_in × r)
    - B ∈ ℝ^(d_out × r)
    - r = rank (8 typical)
    - α = scaling factor (1.0 typical)
    - Parameters: (d_in + d_out) × r << d_in × d_out

Applied to:
    - query projection (to_q)
    - key projection (to_k)
    - value projection (to_v)
    - output projection (to_out)
    
Frozen:
    - All spatial attention layers
    - VAE encoder/decoder
    - Text encoder
    - Scheduler

TIC-FT Buffers:
    Chunk layout:
        [History (5)] → [Buffer (5 noisy)] → [Generated (16)]
    
    Buffer creation:
        For each timestep t in noise_schedule:
            noise_level = t / 1000
            buffer_frame = history_frame + noise_level * gaussian_noise
    
    Blending:
        output = (1 - α) * prev_chunk + α * curr_chunk
        where α increases from 0→1 over overlap region
"""

# ============================================================================
# 4. FILE LOCATIONS
# ============================================================================

"""
Core modules:
    src/models/adapters/lora_temporal_adapter.py       - LoRA adapter
    src/models/adapters/tic_ft_buffer.py               - TIC-FT buffer
    src/models/adapters/__init__.py                    - Module exports

Training:
    scripts/train/train_wan2_2_lora_ticft.py          - Training script

Inference:
    scripts/infer/infer_wan2_2_lora_ticft.py          - Inference script

Documentation:
    instruction.md                                      - Full documentation
"""

# ============================================================================
# 5. COMMAND REFERENCE
# ============================================================================

# Training (10 epochs, 5 buffer frames, rank-8 LoRA):
# bash
# export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0
# python3 scripts/train/train_wan2_2_lora_ticft.py \
#     --manifest data/manifests/train.jsonl \
#     --output_dir checkpoints_lora \
#     --epochs 10 \
#     --batch_size 1 \
#     --lora_rank 8 \
#     --lora_alpha 1.0 \
#     --use_tic_ft_buffers \
#     --tic_ft_buffer_frames 5 \
#     --num_inference_steps 50

# Inference (load LoRA, use TIC-FT buffers):
# bash
# export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0
# python3 scripts/infer/infer_wan2_2_lora_ticft.py \
#     --manifest data/manifests/test.jsonl \
#     --checkpoint checkpoints_lora/lora_epoch10.pt \
#     --output_dir prediction_wan2_2_lora \
#     --use_tic_ft_buffers \
#     --tic_ft_buffer_frames 5 \
#     --num_inference_steps 50 \
#     --max_samples 10

# ============================================================================
# 6. HYPERPARAMETER TUNING
# ============================================================================

"""
LoRA Rank (r):
    - r=4: Very lightweight, fast, lower capacity
    - r=8: Standard, good balance (recommended)
    - r=16: Higher capacity, more parameters, slower
    Rule: Use rank=8 unless OOM or insufficient capacity

LoRA Alpha (α):
    - α=0.5: Weak LoRA effect, more like fine-tuning
    - α=1.0: Full LoRA update (recommended)
    - α=2.0: Strong effect, may diverge
    Rule: Keep α=1.0 for stability

Learning Rate:
    - 1e-4: Too high, unstable gradients
    - 5e-5: Standard for LoRA (recommended)
    - 1e-5: Conservative, slower convergence
    Rule: Start with 5e-5, adjust if needed

Buffer Frames:
    - 3-5: Smooth transitions, minimal overhead
    - 5-10: Longer buffers, smoother but slower
    Rule: Use 5 frames (balance between quality and speed)

Chunk Size:
    - 8-16 frames: Standard, fits in VRAM
    - 16-32 frames: Longer context, higher VRAM
    Rule: Use 16 frames for 480×832

Overlap:
    - 0 frames: No blending, visible discontinuities
    - 4-8 frames: Smooth transitions (recommended)
    - 8+ frames: Expensive, marginal improvement
    Rule: Use 4 frames (25% overlap)
"""

# ============================================================================
# 7. TROUBLESHOOTING
# ============================================================================

"""
OOM Error:
    - Reduce batch_size (already 1)
    - Reduce height/width
    - Reduce chunk_size
    - Use gradient_checkpointing (not yet implemented)
    - Reduce num_inference_steps

Temporal discontinuities:
    - Increase tic_ft_buffer_frames (5→10)
    - Increase tic_ft_overlap (4→8)
    - Try "cosine" noise_schedule instead of "linear"

Poor motion quality:
    - Increase lora_rank (8→16)
    - Increase num_inference_steps (50→100)
    - Increase training epochs
    - Check captions quality

Inference slow:
    - Reduce num_inference_steps
    - Increase tic_ft_chunk_size (16→32)
    - Use smaller resolution
    - Reduce max_samples for testing

LoRA not loading:
    - Check checkpoint path exists
    - Check checkpoint saved successfully
    - Verify LoRA rank matches (8→8)
"""

# ============================================================================
# 8. PAPER REFERENCE
# ============================================================================

"""
"Temporal In-Context Fine-Tuning with Temporal Reasoning for 
Versatile Control of Video Diffusion Models"

Authors: Kim Kinam, Hyung Junha, Choo Jaegul
Paper: https://arxiv.org/pdf/2506.00996

Key contributions:
1. TIC-FT: Efficient few-shot adaptation (10-30 samples)
2. No architectural modifications needed
3. Works with large pre-trained models (CogVideoX-5B, Wan-14B)
4. Outperforms existing baselines in condition fidelity

Key ideas:
- Concatenate condition and target frames along temporal axis
- Insert intermediate buffer frames with increasing noise
- Buffers smooth transitions and align with pretrained dynamics
- No special training, just standard diffusion loss

Results on WTS-like datasets:
- PSNR: 28-32 dB
- SSIM: 0.8-0.9
- LPIPS: 0.1-0.2 (lower is better)
- Can train with as few as 10-30 samples
"""
