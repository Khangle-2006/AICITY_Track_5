"""
Inference for Wan2.2 A14B using pre-computed text embeddings.

Strategy:
1. Pre-compute all prompt_embeds using encode_prompts.py (subprocess, exits cleanly).
2. Delete text_encoder from pipeline (no unused params on GPU), load pre-computed embeddings.
3. Move VAE + transformer to GPU, call pipe() with prompt_embeds, denoise, decode, save.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import gc
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.collate import wts_collate_fn
from src.datasets.wts_dataset import WTSDataset
from src.datasets.transforms import denormalize_image
from src.pipelines.wan_video_pipeline import WanPipelineConfig, build_wan_prompt
from src.models.disentanglement.optical_flow import BackgroundPreservationModule
from src.models.compositing.alpha_blending import ForegroundBackgroundCompositor


def safe_slice(output, T_hist, n_desired):
    """Slice future frames from pipe() output, handling VAE temporal truncation."""
    avail = max(0, output.shape[1] - T_hist)
    actual_n = min(n_desired, avail)
    sliced = output[:, T_hist:T_hist + actual_n]
    sliced = sliced * 2.0 - 1.0
    sliced = sliced.clamp(-1.0, 1.0)
    return sliced, actual_n


def preserve_background(generated_frames, history_frames, background_module, compositor):
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
    if frames.dim() == 4:
        return F.interpolate(frames, size=(output_height, output_width), mode="bilinear", align_corners=False)
    return F.interpolate(frames, size=(output_height, output_width), mode="trilinear", align_corners=False)


def prepare_video_for_wan(frames):
    """Convert [B, T, C, H, W] in [-1,1] -> [B, T, C, H, W] in [0,1] for WAN VideoProcessor."""
    return (frames * 0.5 + 0.5).clamp(0.0, 1.0)


def build_chunk_input(history, homography, C_pad, background_module, use_bg):
    """Build the input tensor for a generation chunk: history + padding."""
    if use_bg:
        bg = background_module.flow_module.warp_future_background(history[:, -1], homography, C_pad)
    else:
        bg = history[:, -1:].repeat(1, C_pad, 1, 1, 1)
    return torch.cat([history, bg], dim=1)


def get_pad(C, T_hist):
    """Compute padding to avoid VAE temporal truncation: total frames = 4*n + 1."""
    return C + ((T_hist + C - 1) % 4)


def blend_overlap(final_frames, chunk, t_start, O, n_write, device):
    """Apply linear alpha blending on the overlap region."""
    O_eff = min(O, n_write)
    if O_eff > 0 and n_write > O_eff:
        alpha = torch.linspace(0.0, 1.0, steps=O_eff, device=device).view(1, O_eff, 1, 1, 1).to(chunk.dtype)
        final_frames[:, t_start:t_start + O_eff] = (
            final_frames[:, t_start:t_start + O_eff] * (1.0 - alpha) + chunk[:, :O_eff] * alpha
        )
        final_frames[:, t_start + O_eff:t_start + n_write] = chunk[:, O_eff:n_write]
    elif n_write > 0:
        final_frames[:, t_start:t_start + n_write] = chunk[:, :n_write]


def run_chunk(pipe, vid_tensor, prompt_embeds, negative_prompt_embeds,
              config, device, history_flow_map=None, lambda_weight=0.0):
    """Run a single denoising chunk, returning output frames [B, T, C, H, W] in [0,1]."""
    try:
        output = pipe(
            video=vid_tensor, prompt=None, negative_prompt=None,
            height=config.height, width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale, strength=config.strength,
            output_type="pt", return_dict=False,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            use_dynamic_cfg=config.use_dynamic_cfg,
            min_guidance_scale=config.min_guidance,
            history_flow_map=history_flow_map, lambda_weight=lambda_weight,
        )[0]
    except Exception:
        return None
    return output


def save_frames(frames, output_dir, scenario_name, output_height, output_width):
    sample_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(sample_dir, exist_ok=True)
    resized = resize_for_submission(frames[0], output_height, output_width)
    for i in range(resized.size(0)):
        torchvision.utils.save_image(
            denormalize_image(resized[i]), os.path.join(sample_dir, f"{i}.png")
        )


def save_ground_truth(batch, output_dir, scenario_name, output_height, output_width):
    if "future_frames" not in batch:
        return
    gt_frames = denormalize_image(batch["future_frames"])[0]
    gt_resized = resize_for_submission(gt_frames, output_height, output_width)
    gt_dir = os.path.join(output_dir, f"{scenario_name}_gt")
    os.makedirs(gt_dir, exist_ok=True)
    for i in range(gt_resized.size(0)):
        torchvision.utils.save_image(gt_resized[i], os.path.join(gt_dir, f"{i}.png"))


def cleanup_gpu(pipe):
    pipe.transformer = pipe.transformer.to("cpu")
    pipe.vae = pipe.vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def infer(args):
    print("=" * 72)
    print(f"Wan2.2 V2V Inference | Model: {args.model_id}")
    print(f"Res: {args.height}x{args.width} -> {args.output_width}x{args.output_height}")
    print(f"Steps: {args.num_inference_steps}, CFG: {args.guidance_scale}")
    print("=" * 72)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    dataset = WTSDataset(
        args.manifest, is_train=False,
        size=(args.height, args.width),
        max_history_frames=args.h,
        history_strategy=args.history_strategy,
    )
    if args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]

    for i, sample in enumerate(dataset.samples):
        if "scenario_name" not in sample:
            sample["scenario_name"] = f"sample_{i}"

    if args.world_size > 1:
        samples_per_rank = len(dataset.samples) // args.world_size
        start_idx = args.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if args.rank < args.world_size - 1 else len(dataset.samples)
        dataset.samples = dataset.samples[start_idx:end_idx]
        print(f"[Rank {args.rank}/{args.world_size}] Partitioned dataset: processing indices {start_idx} to {end_idx} ({len(dataset.samples)} samples).")
    else:
        print(f"Loaded {len(dataset)} samples")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=wts_collate_fn, num_workers=args.num_workers,
    )

    background_module = BackgroundPreservationModule(
        use_raft=False, ransac_threshold=args.ransac_threshold,
        motion_threshold=args.motion_threshold, flow_stride=args.flow_stride,
    ).to(device)
    compositor = ForegroundBackgroundCompositor(
        use_depth_aware=False, motion_mask_weight=0.5,
    ).to(device)

    config = WanPipelineConfig(
        model_id=args.model_id, torch_dtype=args.torch_dtype,
        height=args.height, width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, min_guidance=args.min_guidance,
        strength=args.strength, flow_shift=args.flow_shift,
        max_sequence_length=args.max_sequence_length, chunk_size=args.chunk_size,
        output_width=args.output_width, output_height=args.output_height,
        use_background_preservation=args.use_background_preservation,
        use_alpha_compositing=args.use_alpha_compositing,
        use_dynamic_cfg=args.use_dynamic_cfg,
    )

    from src.pipelines.wan2_2_v2v_pipeline import Wan2_2V2VPipeline as HFP
    from diffusers import AutoencoderKLWan

    print("Loading pipeline...")
    vae_dtype = torch.float16 if args.vae_dtype == "float16" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=vae_dtype)
    if vae_dtype is torch.float32:
        vae.enable_tiling()
        vae.enable_slicing()
    pipe = HFP.from_pretrained(args.model_id, torch_dtype=torch.float16, vae=vae)
    pipe.text_encoder = None
    gc.collect()
    print(f"  Pipeline loaded, TE offloaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    embeds_dir = args.prompt_embeds_dir
    print(f"Using pre-computed embeddings from: {embeds_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    sample_idx = 0
    start_time = time.time()
    output = None

    for batch in dataloader:
        if args.max_samples and sample_idx >= args.max_samples:
            break

        scenario_name = batch["scenario_name"][0]
        history_frames = batch["history_frames"].to(device)
        phase_label = batch["phase_label"][0].to(device)
        sample_type = batch["sample_type"][0]
        caption = batch["merged_caption"][0]
        target_frames = int(batch.get("frame_length", [args.chunk_size])[0])
        prompt = build_wan_prompt(caption, phase_label, sample_type)

        embed_path = os.path.join(embeds_dir, f"{scenario_name}.pt")
        if not os.path.exists(embed_path):
            print(f"  WARNING: missing embeddings for {scenario_name}, skipping")
            sample_idx += 1
            continue
        embeds = torch.load(embed_path, map_location="cpu", weights_only=True)
        prompt_embeds = embeds["prompt_embeds"].to(device, dtype=dtype)
        negative_prompt_embeds = embeds["negative_prompt_embeds"].to(device, dtype=dtype)
        del embeds

        pipe.vae = pipe.vae.to(device)
        pipe.transformer = pipe.transformer.to(device)
        print(f"  [GPU] models loaded: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

        T_hist = history_frames.shape[1]
        max_chunk = min(args.chunk_size, target_frames)

        if args.use_background_preservation:
            homography, _, history_flow_map = background_module.flow_module(history_frames)
        else:
            homography = None
            history_flow_map = None

        final_frames = torch.zeros(
            1, target_frames, 3, config.height, config.width,
            device=history_frames.device, dtype=history_frames.dtype,
        )

        L = 0
        chunk_idx = 0
        while L < target_frames:
            C = min(target_frames - L, max_chunk)
            C_pad = get_pad(C, T_hist)

            if L == 0:
                curr_history = history_frames
            else:
                O = min(args.overlap_frames, L)
                t_start = L - O
                seq = torch.cat([history_frames, final_frames[:, :L]], dim=1)
                curr_history = seq[:, t_start:t_start + T_hist]

            frames_input = build_chunk_input(
                curr_history, homography, C_pad,
                background_module, args.use_background_preservation,
            )
            vid_tensor = prepare_video_for_wan(frames_input)

            output = run_chunk(pipe, vid_tensor, prompt_embeds, negative_prompt_embeds,
                               config, device, history_flow_map, args.lambda_weight)
            if output is None:
                print(f"  ERROR in pipe() chunk {chunk_idx}")
                break

            chunk, n_avail = safe_slice(output, T_hist, C)

            if args.use_background_preservation:
                chunk = preserve_background(chunk, curr_history, background_module, compositor)

            if L == 0:
                final_frames[:, :n_avail] = chunk
                L = n_avail
            else:
                n_write = min(n_avail, target_frames - t_start)
                blend_overlap(final_frames, chunk, t_start, O, n_write, device)
                L = t_start + n_write

            chunk_idx += 1

        composited = final_frames[:, :target_frames]
        save_frames(composited, args.output_dir, scenario_name, args.output_height, args.output_width)
        save_ground_truth(batch, args.output_dir, scenario_name, args.output_height, args.output_width)

        sample_idx += 1
        print(f"  [{sample_idx}/{args.max_samples or len(dataset)}] {scenario_name} -> {os.path.join(args.output_dir, scenario_name)}")

        prompt_embeds = None
        negative_prompt_embeds = None
        output = None
        cleanup_gpu(pipe)
        if not args.keep_on_gpu:
            print(f"  [GPU] after cleanup: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    elapsed = time.time() - start_time
    n = max(sample_idx, 1)
    print(f"\nDone. {sample_idx} samples in {elapsed:.1f}s ({elapsed/n:.1f}s/sample)")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 V2V Inference")
    parser.add_argument("--manifest", type=str, default="/workspace/wts_baseline/data/manifests/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="/workspace/wts_baseline/prediction_wan2.2_val_5")
    parser.add_argument("--prompt_embeds_dir", type=str, default="/tmp/prompt_embeds")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--output_width", type=int, default=1280)
    parser.add_argument("--output_height", type=int, default=720)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--min_guidance", type=float, default=1.5)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=81)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--history_strategy", choices=["tail", "dense_tail", "uniform"], default="dense_tail")
    parser.add_argument("--use_background_preservation", action="store_true", default=True)
    parser.add_argument("--use_alpha_compositing", action="store_true", default=True)
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=True)
    parser.add_argument("--ransac_threshold", type=float, default=1.0)
    parser.add_argument("--motion_threshold", type=float, default=0.05)
    parser.add_argument("--flow_stride", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overlap_frames", type=int, default=2)
    parser.add_argument("--lambda_weight", type=float, default=0.1)
    parser.add_argument("--keep_on_gpu", action="store_true", default=False,
                        help="Keep VAE + transformer on GPU between samples (avoids CPU offload overhead). "
                             "Requires A6000 (50GB) or similar. Not suitable for A100 (40GB).")
    parser.add_argument("--vae_dtype", type=str, default="float32", choices=["float32", "float16"],
                        help="VAE dtype. Use float16 on A6000 to save ~1.5GB VRAM. "
                             "float32 enables tiling (lower memory).")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    args = parser.parse_args()
    if not os.path.exists(args.manifest):
        alt = os.path.join("/data/Track_5/WTS_dataset/WTS_dataset", args.manifest)
        if os.path.exists(alt):
            args.manifest = alt
    infer(args)


if __name__ == "__main__":
    main()
