"""
Inference script for WAN2.2 with LoRA Temporal Adaptation + TIC-FT Buffer

This script demonstrates how to use a fine-tuned LoRA adapter with TIC-FT
for generating long video sequences (e.g., 81 frames) with smooth rolling-window
inference.

Based on: https://arxiv.org/pdf/2506.00996
Title: "Temporal In-Context Fine-Tuning with Temporal Reasoning for Versatile 
Control of Video Diffusion Models"

Usage:
    export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0
    python3 scripts/infer/infer_wan2_2_lora_ticft.py \
        --manifest data/manifests/test.jsonl \
        --checkpoint checkpoints_lora/lora_epoch10.pt \
        --output_dir prediction_wan2_2_lora \
        --use_tic_ft_buffers \
        --height 480 --width 832 \
        --output_height 1280 --output_width 720 \
        --max_samples 10
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusers import WanVideoToVideoPipeline
from src.datasets.collate import wts_collate_fn
from src.datasets.wts_dataset import WTSDataset
from src.datasets.transforms import denormalize_image
from src.models.adapters import (
    apply_lora_to_pipeline,
    TICFTBufferManager,
    RollingWindowInferenceEngine,
)
from src.pipelines.wan_video_pipeline import build_wan_prompt
from src.models.disentanglement.optical_flow import BackgroundPreservationModule
from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor


def preserve_background(generated_frames, history_frames, background_module, compositor):
    """Apply background preservation and alpha compositing."""
    B, T_gen, C, H, W = generated_frames.shape
    background, motion_mask = background_module(history_frames, T_gen)
    foreground = generated_frames.clone()
    alpha_mask = torch.ones(B, 1, T_gen, H, W, device=generated_frames.device)
    composited = compositor(
        foreground.permute(0, 2, 1, 3, 4),
        background.permute(0, 2, 1, 3, 4),
        alpha_mask,
        motion_mask=motion_mask.unsqueeze(2) if motion_mask.dim() == 4 else motion_mask,
    )
    return composited.permute(0, 2, 1, 3, 4)


def resize_for_submission(frames, output_height, output_width):
    """Resize frames to submission format."""
    # input: [B, T, C, H, W]
    B, T, C, H, W = frames.shape
    # fold T into batch dimension
    frames_folded = frames.reshape(B * T, C, H, W)
    # bilinear interpolate spatially
    resized = F.interpolate(
        frames_folded, 
        size=(output_height, output_width), 
        mode="bilinear", 
        align_corners=False
    )
    # unfold back
    return resized.reshape(B, T, C, output_height, output_width)


def _prepare_video_for_wan(frames: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] to [0,1] for WAN VideoProcessor."""
    return (frames * 0.5 + 0.5).clamp(0.0, 1.0)


def setup_inference_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference for WAN2.2 with LoRA + TIC-FT"
    )
    
    # Data
    parser.add_argument("--manifest", type=str, required=True, help="Test manifest path")
    parser.add_argument("--output_dir", type=str, default="prediction_wan2_2_lora", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Model
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-1.3B-Diffusers")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    
    # LoRA
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    
    # TIC-FT
    parser.add_argument("--use_tic_ft_buffers", action="store_true", help="Use TIC-FT buffer frames")
    parser.add_argument("--tic_ft_buffer_frames", type=int, default=5)
    parser.add_argument("--tic_ft_chunk_size", type=int, default=16)
    parser.add_argument("--tic_ft_overlap", type=int, default=4)
    parser.add_argument("--tic_ft_noise_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    
    # Inference
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--output_height", type=int, default=1280)
    parser.add_argument("--output_width", type=int, default=720)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--min_guidance", type=float, default=1.0)
    
    # Processing
    parser.add_argument("--use_background_preservation", action="store_true")
    parser.add_argument("--use_alpha_compositing", action="store_true")
    parser.add_argument("--ransac_threshold", type=float, default=1.0)
    parser.add_argument("--motion_threshold", type=float, default=0.05)
    parser.add_argument("--flow_stride", type=int, default=16)
    
    # Distributed
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    
    return parser.parse_args()


@torch.no_grad()
def infer(args):
    """Run inference."""
    print("=" * 80)
    print(f"🎬 WAN2.2 Inference with LoRA + TIC-FT")
    print(f"Model: {args.model_id}")
    print(f"Resolution: {args.height}x{args.width} → {args.output_width}x{args.output_height}")
    print(f"LoRA checkpoint: {args.checkpoint}")
    print(f"TIC-FT enabled: {args.use_tic_ft_buffers}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    
    # Load pipeline
    print("\n📦 Loading WAN2.2 pipeline...")
    pipeline = WanVideoToVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    # Apply LoRA adaptation first (before offloading)
    print("🔧 Applying LoRA adapter...")
    adapter_module = apply_lora_to_pipeline(
        pipeline,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        enable_training=False,  # Inference mode
    )
    
    # Patch VAE encode/decode to handle float32 inputs from diffusers pipeline
    print("🩹 Patching VAE encode/decode for dtype compatibility...")
    original_vae_encode = pipeline.vae.encode
    original_vae_decode = pipeline.vae.decode
    
    def patched_vae_encode(x, *args, **kwargs):
        vae_dtype = next(pipeline.vae.parameters()).dtype
        return original_vae_encode(x.to(dtype=vae_dtype), *args, **kwargs)
        
    def patched_vae_decode(z, *args, **kwargs):
        vae_dtype = next(pipeline.vae.parameters()).dtype
        return original_vae_decode(z.to(dtype=vae_dtype), *args, **kwargs)
        
    pipeline.vae.encode = patched_vae_encode
    pipeline.vae.decode = patched_vae_decode
    
    # Enable CPU offloading to fit in VRAM
    print("⚡ Enabling model CPU offload...")
    pipeline.enable_model_cpu_offload()
    
    # Load LoRA weights
    if os.path.exists(args.checkpoint):
        print(f"📥 Loading LoRA weights from {args.checkpoint}...")
        adapter_module.load_lora_weights(args.checkpoint, device=device)
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print("   Proceeding with uninitialized LoRA (random initialization)")
    
    # Setup TIC-FT if enabled
    tic_ft_manager = None
    engine = None
    if args.use_tic_ft_buffers:
        tic_ft_manager = TICFTBufferManager(
            buffer_frames=args.tic_ft_buffer_frames,
            chunk_size=args.tic_ft_chunk_size,
            overlap_frames=args.tic_ft_overlap,
            noise_schedule=args.tic_ft_noise_schedule,
        )
        engine = RollingWindowInferenceEngine(
            pipeline=pipeline,
            tic_ft_manager=tic_ft_manager,
            device=device,
        )
        print(f"✓ TIC-FT Buffer Manager & Rolling Window Inference Engine initialized")
        print(f"  - Buffer frames: {args.tic_ft_buffer_frames}")
        print(f"  - Chunk size: {args.tic_ft_chunk_size}")
        print(f"  - Overlap: {args.tic_ft_overlap}")
    
    # Setup background preservation modules
    background_module = None
    compositor = None
    if args.use_background_preservation:
        background_module = BackgroundPreservationModule(
            use_raft=False,
            ransac_threshold=args.ransac_threshold,
            motion_threshold=args.motion_threshold,
            flow_stride=args.flow_stride,
        ).to(device)
        compositor = ForegroundBackgroundCompositor(
            use_depth_aware=False,
            motion_mask_weight=0.5,
        ).to(device)
        print("✓ Background preservation enabled")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.manifest}...")
    dataset = WTSDataset(
        args.manifest,
        is_train=False,
        size=(args.height, args.width),
        max_history_frames=5,
        history_strategy="first_n",
    )
    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
    
    # Setup scenario names
    for i, sample in enumerate(dataset.samples):
        if "scenario_name" not in sample:
            sample["scenario_name"] = f"sample_{i}"
    
    # Distributed partition
    if args.world_size > 1:
        samples_per_rank = len(dataset.samples) // args.world_size
        start_idx = args.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if args.rank < args.world_size - 1 else len(dataset.samples)
        dataset.samples = dataset.samples[start_idx:end_idx]
        print(f"[Rank {args.rank}/{args.world_size}] Processing {len(dataset.samples)} samples")
    else:
        print(f"Loaded {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=wts_collate_fn,
        num_workers=args.num_workers,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inference loop
    total_samples = len(dataloader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        scenario_name = batch["scenario_name"][0]
        print(f"\n[{batch_idx + 1}/{total_samples}] Processing {scenario_name}...")
        
        history_frames = batch["history_frames"].to(device, dtype=dtype)  # [1, 5, C, H, W]
        T_hist = history_frames.shape[1]
        
        # Build prompt and encode it online
        caption = batch["merged_caption"][0]
        phase_label = batch["phase_label"][0].item() if isinstance(batch["phase_label"][0], torch.Tensor) else batch["phase_label"][0]
        sample_type = batch["sample_type"][0]
        prompt = build_wan_prompt(caption, phase_label, sample_type)
        
        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "deformed, disfigured, misshapen, still picture, messy background"
        )
        
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=args.guidance_scale > 1.0,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=device,
            )
            prompt_embeds = prompt_embeds.to(dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)
        
        chunk_start = time.time()
        
        # Generate frames using TIC-FT rolling window
        with torch.no_grad():
            if args.use_tic_ft_buffers and engine is not None:
                # Use rolling window with TIC-FT buffers
                try:
                    target_frames = int(batch["frame_length"][0]) if "frame_length" in batch else 81
                    history_frames_01 = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)
                    generated_frames = engine.generate_long_video(
                        history_frames=history_frames_01,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        total_frames=target_frames,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                    )  # [1, T_hist + target_frames, C, H, W]
                    
                    # Slice generated frames (trim off history frames from the beginning)
                    generated_frames = generated_frames[:, T_hist:]
                    # Convert to [-1, 1] range to match post-processing expectation
                    generated_frames = generated_frames * 2.0 - 1.0
                except Exception as e:
                    print(f"⚠️  Error during generation: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Standard inference without TIC-FT
                history_frames_01 = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)
                generated_frames = pipeline(
                    video=history_frames_01,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                ).frames
                
                # Convert to [-1, 1] range
                generated_frames = generated_frames * 2.0 - 1.0
        
        chunk_time = time.time() - chunk_start
        
        # Post-processing
        generated_frames = _prepare_video_for_wan(generated_frames)
        
        # Background preservation
        if args.use_background_preservation and background_module is not None:
            try:
                generated_frames = preserve_background(
                    generated_frames,
                    history_frames,
                    background_module,
                    compositor,
                )
            except Exception as e:
                print(f"⚠️  Background preservation failed: {e}")
        
        # Resize to submission format
        generated_frames = resize_for_submission(
            generated_frames,
            args.output_height,
            args.output_width,
        )
        
        # Save frames
        output_subdir = os.path.join(args.output_dir, scenario_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        T = generated_frames.shape[1]
        for t in range(T):
            frame = generated_frames[0, t]  # [C, H, W]
            frame_uint8 = (frame * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            frame_pil = torchvision.transforms.ToPILImage()(torch.from_numpy(frame_uint8).permute(2, 0, 1))
            frame_path = os.path.join(output_subdir, f"{t:04d}.png")
            frame_pil.save(frame_path)
            
        # Save ground truth frames if present in batch for evaluation
        if "future_frames" in batch:
            gt_frames = denormalize_image(batch["future_frames"])[0]  # [T_fut, C, H, W]
            gt_resized = resize_for_submission(
                gt_frames.unsqueeze(0), 
                args.output_height, 
                args.output_width
            )[0]
            gt_dir = os.path.join(args.output_dir, f"{scenario_name}_gt")
            os.makedirs(gt_dir, exist_ok=True)
            for t in range(gt_resized.size(0)):
                frame = gt_resized[t]
                frame_uint8 = (frame * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                frame_pil = torchvision.transforms.ToPILImage()(torch.from_numpy(frame_uint8).permute(2, 0, 1))
                frame_path = os.path.join(gt_dir, f"{t:04d}.png")
                frame_pil.save(frame_path)
        
        elapsed = chunk_time
        print(f"   ✓ Generated {T} frames in {elapsed:.1f}s ({T/elapsed:.1f} fps)")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✓ Inference completed in {total_time:.1f}s")
    print(f"Frames saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    args = setup_inference_args()
    infer(args)
