"""
Training script for WAN2.2 with LoRA Temporal Adaptation + TIC-FT Buffer

This script demonstrates how to fine-tune WAN2.2 on WTS dataset using:
1. LoRA adapters on temporal attention layers (not spatial)
2. Frozen spatial layers and VAE
3. TIC-FT buffer insertion for smooth rolling-window generation

Based on: https://arxiv.org/pdf/2506.00996
Title: "Temporal In-Context Fine-Tuning with Temporal Reasoning for Versatile 
Control of Video Diffusion Models"

Usage:
    export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0
    python3 scripts/train/train_wan2_2_lora_ticft.py \
        --manifest data/manifests/train.jsonl \
        --output_dir checkpoints_lora \
        --epochs 10 \
        --batch_size 1 \
        --lora_rank 8 \
        --lora_alpha 1.0 \
        --tic_ft_buffer_frames 5 \
        --use_tic_ft_buffers
"""

import argparse
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
import json
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from diffusers import WanVideoToVideoPipeline, AutoencoderKLWan, WanTransformer3DModel
from src.datasets.collate import wts_collate_fn
from src.datasets.wts_dataset import WTSDataset
from src.datasets.transforms import denormalize_image
from src.models.adapters import (
    apply_lora_to_pipeline,
    TICFTBufferManager,
    RollingWindowInferenceEngine,
)
from src.pipelines.wan_video_pipeline import build_wan_prompt


def create_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load WAN2.2 pipeline and return components."""
    print(f"📦 Loading {model_id}...")
    
    pipeline = WanVideoToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    
    return pipeline


def setup_training_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune WAN2.2 with LoRA + TIC-FT for WTS dataset"
    )
    
    # Data
    parser.add_argument("--manifest", type=str, required=True, help="Training manifest path")
    parser.add_argument("--val_manifest", type=str, default=None, help="Validation manifest path (optional)")
    parser.add_argument("--output_dir", type=str, default="checkpoints_lora", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for debugging")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for LoRA parameters")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Model
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-1.3B-Diffusers")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # TIC-FT
    parser.add_argument("--use_tic_ft_buffers", action="store_true", help="Use TIC-FT buffer frames")
    parser.add_argument("--tic_ft_buffer_frames", type=int, default=5, help="Number of buffer frames")
    parser.add_argument("--tic_ft_chunk_size", type=int, default=16, help="Frames per chunk")
    parser.add_argument("--tic_ft_overlap", type=int, default=4, help="Frame overlap between chunks")
    
    # Inference
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    
    # Logging
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    
    # Hardware
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    
    return parser.parse_args()


def save_checkpoint(
    adapter_module,
    optimizer,
    epoch: int,
    step: int,
    output_dir: str,
):
    """Save LoRA weights and training state."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LoRA weights
    lora_path = os.path.join(output_dir, f"lora_epoch{epoch}_step{step}.pt")
    adapter_module.save_lora_weights(lora_path)
    
    # Save optimizer state and metadata
    state = {
        "epoch": epoch,
        "step": step,
        "optimizer": optimizer.state_dict(),
    }
    state_path = os.path.join(output_dir, f"train_state_epoch{epoch}_step{step}.pt")
    torch.save(state, state_path)
    
    print(f"✓ Checkpoint saved: {lora_path}")


def train_step(
    batch,
    pipeline,
    adapter_module,
    optimizer,
    args,
    device: torch.device,
    dtype: torch.dtype,
):
    """Single training step."""
    if "future_frames" not in batch:
        print("⚠️  Skipping batch with no future_frames")
        return None

    # 1. Extract and preprocess history and future frames (keep on CPU initially)
    history_frames = batch["history_frames"].to(dtype=dtype)  # [B, T_hist, C, H, W]
    future_frames = batch["future_frames"].to(dtype=dtype)    # [B, T_fut, C, H, W]
    
    B, T_hist, C, H, W = history_frames.shape
    B, T_fut, C, H_fut, W_fut = future_frames.shape

    # Denormalize from [-1, 1] to [0, 1] for pipeline/VAE preprocessing
    history_frames_01 = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)
    future_frames_01 = (future_frames * 0.5 + 0.5).clamp(0.0, 1.0)

    # 2. Setup progressive noise buffers if enabled (run on CPU/device as needed)
    if args.use_tic_ft_buffers:
        # Insert buffers: history -> buffer -> future
        tic_ft_manager = TICFTBufferManager(
            buffer_frames=args.tic_ft_buffer_frames,
            chunk_size=args.tic_ft_chunk_size,
            overlap_frames=args.tic_ft_overlap,
        )
        concatenated, _, buffer_mask = tic_ft_manager.insert_buffer_frames(
            condition_frames=history_frames_01,
            target_frames=future_frames_01,
            device=torch.device("cpu"),
        )
        T_buf = args.tic_ft_buffer_frames
    else:
        concatenated = torch.cat([history_frames_01, future_frames_01], dim=1)
        T_buf = 0
        
    T_total = concatenated.shape[1]

    # 3. Encode prompt online (dynamically offload Text Encoder to CPU to save GPU memory)
    prompts = [
        build_wan_prompt(caption, phase_label.item(), sample_type)
        for caption, phase_label, sample_type in zip(batch["merged_caption"], batch["phase_label"], batch["sample_type"])
    ]
    
    with torch.no_grad():
        pipeline.text_encoder = pipeline.text_encoder.to(device)
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        # Move text encoder back to CPU and empty cache
        pipeline.text_encoder = pipeline.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # 4. Preprocess video for VAE and encode to latent space (dynamically load/offload VAE to CPU)
    # Preprocess video on CPU
    video_tensor = pipeline.video_processor.preprocess_video(concatenated, height=H, width=W)
    
    with torch.no_grad():
        # Move VAE to GPU
        pipeline.vae = pipeline.vae.to(device)
        video_tensor = video_tensor.to(device, dtype=pipeline.vae.dtype)
        
        # Encode
        latents_total = pipeline.vae.encode(video_tensor).latent_dist.sample()
        
        # Normalize
        latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype=latents_total.dtype)
        latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype=latents_total.dtype)
        latents_total = (latents_total - latents_mean) * latents_std
        latents_total = latents_total.to(dtype=dtype)
        
        # Move VAE back to CPU and empty cache
        pipeline.vae = pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

    # 5. Split condition + buffer vs target in latent space
    # VAE temporal scale factor is 4, so T_lat = (T - 1) // 4 + 1
    T_lat_cond = (T_hist + T_buf - 1) // 4 + 1
    
    latents_cond = latents_total[:, :, :T_lat_cond]
    latents_target = latents_total[:, :, T_lat_cond:]
    
    if latents_target.shape[2] == 0:
        print(f"⚠️  Target latents are empty (T_lat_total={latents_total.shape[2]}, T_lat_cond={T_lat_cond})")
        return None

    # 6. Sample random timestep t for flow-matching target
    t = torch.rand((B,), device=device, dtype=dtype)
    t_expanded = t.view(-1, 1, 1, 1, 1)
    
    noise = torch.randn_like(latents_target)
    noisy_latents_target = (1.0 - t_expanded) * latents_target + t_expanded * noise
    
    # 7. Concatenate condition/buffer and noisy target
    latents_in = torch.cat([latents_cond, noisy_latents_target], dim=2)
    
    # 8. Transformer forward pass (timesteps in [0, 1000])
    timestep_input = t * 1000.0
    
    pred = pipeline.transformer(
        hidden_states=latents_in,
        timestep=timestep_input,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]
    
    # Slice the prediction matching the target latents
    pred_target = pred[:, :, T_lat_cond:]
    
    # Ground truth velocity target
    target_velocity = noise - latents_target
    
    # 9. Compute MSE loss
    loss = F.mse_loss(pred_target.float(), target_velocity.float(), reduction="mean")
    
    return loss


@torch.no_grad()
def validation_step(
    pipeline,
    adapter_module,
    tic_ft_manager: Optional[TICFTBufferManager],
    batch,
    args,
    device: torch.device,
    dtype: torch.dtype,
):
    """Run validation with TIC-FT buffer generation."""
    adapter_module.disable_lora()  # Inference mode

    if "future_frames" not in batch:
        print("  ⚠️ Skipping validation batch with no future_frames")
        adapter_module.enable_lora()
        return None

    history_frames = batch["history_frames"].to(device, dtype=dtype)
    future_frames = batch["future_frames"].to(device, dtype=dtype)
    B, T_hist, C, H, W = history_frames.shape
    T_fut = future_frames.shape[1]

    # Build prompt
    caption = batch["merged_caption"][0]
    phase_label = batch["phase_label"][0].item() if isinstance(batch["phase_label"][0], torch.Tensor) else batch["phase_label"][0]
    sample_type = batch["sample_type"][0]
    prompt = build_wan_prompt(caption, phase_label, sample_type)

    # Encode prompt
    pipeline.text_encoder = pipeline.text_encoder.to(device)
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=(
            "Bright tones, overexposed, static, blurred details, subtitles, "
            "worst quality, low quality, JPEG compression residue"
        ),
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)
    pipeline.text_encoder = pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    history_frames_01 = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)

    if tic_ft_manager is not None:
        engine = RollingWindowInferenceEngine(
            pipeline=pipeline,
            tic_ft_manager=tic_ft_manager,
            device=device,
        )
        generated_frames = engine.generate_long_video(
            history_frames=history_frames_01,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            total_frames=T_fut,
            guidance_scale=7.5,
            num_inference_steps=args.num_inference_steps,
        )
        generated_frames = generated_frames[:, T_hist:]
        generated_frames = generated_frames * 2.0 - 1.0
    else:
        history_frames_01 = (history_frames * 0.5 + 0.5).clamp(0.0, 1.0)
        generated_frames = pipeline(
            video=history_frames_01,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=7.5,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
        ).frames
        generated_frames = generated_frames * 2.0 - 1.0

    # Compute validation metrics (MSE, PSNR)
    T_min = min(generated_frames.shape[1], T_fut)
    gen = generated_frames[:, :T_min]
    gt = future_frames[:, :T_min]
    mse = F.mse_loss(gen, gt)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8))
    print(f"  📊 Validation: MSE={mse.item():.6f}, PSNR={psnr.item():.2f} dB")

    adapter_module.enable_lora()
    return {"mse": mse.item(), "psnr": psnr.item()}


def main(args):
    print("=" * 80)
    print("🎓 WAN2.2 LoRA + TIC-FT Training")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Using TIC-FT buffers: {args.use_tic_ft_buffers}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    
    # Load pipeline (keep VAE and Text Encoder on CPU by default)
    pipeline = create_pipeline(args.model_id, device, dtype)
    # Move transformer to GPU (where it will remain for training)
    pipeline.transformer = pipeline.transformer.to(device)
    
    # Enable gradient checkpointing to save VRAM
    if hasattr(pipeline.transformer, "enable_gradient_checkpointing"):
        print("⚡ Enabling gradient checkpointing on transformer...")
        pipeline.transformer.enable_gradient_checkpointing()
    else:
        print("⚠️ Transformer does not support enable_gradient_checkpointing")
    
    # Apply LoRA adaptation
    print("\n🔧 Applying LoRA adaptation to temporal attention layers...")
    adapter_module = apply_lora_to_pipeline(
        pipeline,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        enable_training=True,
    )
    adapter_module = adapter_module.to(device)
    
    # Setup optimizer (only LoRA parameters)
    trainable_params = [p for p in adapter_module.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    print(f"\n📊 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.manifest}...")
    dataset = WTSDataset(
        args.manifest,
        is_train=True,
        size=(args.height, args.width),
        max_history_frames=5,
        history_strategy="first_n",
    )
    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=wts_collate_fn,
    )
    
    # Setup TIC-FT manager if using buffers
    tic_ft_manager = None
    if args.use_tic_ft_buffers:
        tic_ft_manager = TICFTBufferManager(
            buffer_frames=args.tic_ft_buffer_frames,
            chunk_size=args.tic_ft_chunk_size,
            overlap_frames=args.tic_ft_overlap,
        )
        print(f"✓ TIC-FT buffer manager created")
    
    # Load validation dataset if a separate manifest is provided
    val_loader = None
    val_tfm = None
    if args.val_manifest and os.path.exists(args.val_manifest):
        val_dataset = WTSDataset(
            args.val_manifest,
            is_train=False,
            size=(args.height, args.width),
            max_history_frames=5,
            history_strategy="first_n",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=wts_collate_fn,
            num_workers=args.num_workers,
        )
        print(f"✓ Validation set loaded: {len(val_dataset)} samples")

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        adapter_module.enable_lora()  # Training mode
        total_loss = 0.0
        num_batches = 0
        accum_count = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(
                batch,
                pipeline,
                adapter_module,
                optimizer,
                args,
                device,
                dtype,
            )
            
            if loss is None:
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            accum_count += 1
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1
            
            if accum_count >= args.gradient_accumulation_steps:
                # Gradient clipping
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0
                
                if global_step % args.log_every == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Step {global_step}: loss = {avg_loss:.4f}")
                
                if global_step % args.save_every == 0:
                    save_checkpoint(
                        adapter_module,
                        optimizer,
                        epoch,
                        global_step,
                        args.output_dir,
                    )
        
        # Flush remaining accumulated gradients
        if accum_count > 0:
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Save epoch checkpoint
        save_checkpoint(
            adapter_module,
            optimizer,
            epoch + 1,
            global_step,
            args.output_dir,
        )
        
        print(f"\n✓ Epoch {epoch + 1} completed. Avg loss: {total_loss / max(num_batches, 1):.4f}")

        # Run validation after each epoch
        if val_loader is not None:
            print(f"\n  Running validation on {len(val_loader)} samples...")
            val_mse_list = []
            val_psnr_list = []
            for val_idx, val_batch in enumerate(val_loader):
                val_result = validation_step(
                    pipeline,
                    adapter_module,
                    tic_ft_manager,
                    val_batch,
                    args,
                    device,
                    dtype,
                )
                if val_result is not None:
                    val_mse_list.append(val_result["mse"])
                    val_psnr_list.append(val_result["psnr"])
            if val_mse_list:
                avg_val_mse = sum(val_mse_list) / len(val_mse_list)
                avg_val_psnr = sum(val_psnr_list) / len(val_psnr_list)
                print(f"  📊 Validation results: MSE={avg_val_mse:.6f}, PSNR={avg_val_psnr:.2f} dB")
    
    print("\n" + "=" * 80)
    print("✓ Training completed!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    args = setup_training_args()
    main(args)
