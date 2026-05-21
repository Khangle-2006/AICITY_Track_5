"""
SOTA Inference entry point for Wan2.1-based video-to-video pipeline.
AI City Challenge 2026 Track 5 - Generative Traffic Video Forecasting.

Integrates:
  - Wan2.1 3D Causal VAE + Flow-Matching DiT
  - Optical flow background disentanglement
  - Adaptive alpha blending compositing
  - Dynamic CFG scheduling
  - Autoregressive sliding window generation
"""

import argparse
import os
import sys
import time

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.collate import wts_collate_fn
from src.datasets.wts_dataset import WTSDataset
from src.pipelines.wan_video_pipeline import WanPipelineConfig, WanSOTAPipeline


def infer(args):
    print("=" * 72)
    print("Wan2.1 SOTA Video-to-Video Inference - AI City 2026 Track 5")
    print(f"Model: {args.model_id}")
    print(f"Resolution: {args.height}x{args.width} -> {args.output_width}x{args.output_height}")
    print(f"CFG: {args.guidance_scale} (min {args.min_guidance}), Steps: {args.num_inference_steps}")
    print(f"Background preservation: {args.use_background_preservation}")
    print(f"Dynamic CFG: {args.use_dynamic_cfg}")
    print("=" * 72)

    dataset = WTSDataset(
        args.manifest,
        is_train=False,
        size=(args.height, args.width),
        max_history_frames=args.max_history_frames or args.h,
        history_strategy=args.history_strategy,
    )
    print(f"Loaded {len(dataset)} samples for inference.")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=wts_collate_fn, num_workers=args.num_workers,
    )

    config = WanPipelineConfig(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        min_guidance=args.min_guidance,
        strength=args.strength,
        flow_shift=args.flow_shift,
        max_sequence_length=args.max_sequence_length,
        chunk_size=args.chunk_size,
        output_width=args.output_width,
        output_height=args.output_height,
        use_background_preservation=args.use_background_preservation,
        use_alpha_compositing=args.use_alpha_compositing,
        use_dynamic_cfg=args.use_dynamic_cfg,
        use_step_caching=args.use_step_caching,
        overlap_frames=args.overlap_frames,
    )
    pipeline = WanSOTAPipeline(config)

    os.makedirs(args.output_dir, exist_ok=True)
    sample_idx = 0
    start_time = time.time()

    for batch in dataloader:
        if sample_idx >= args.max_samples:
            break

        pred_frames = pipeline.generate_batch(batch)
        sample_dir = pipeline.save_batch(batch, pred_frames, args.output_dir)
        sample_idx += 1

        target_frames = batch.get("frame_length", [args.chunk_size])[0]
        original_k = batch.get("original_history_length", [batch["history_frames"].size(1)])[0]
        used_k = batch["history_frames"].size(1)
        sample_type = batch.get("sample_type", ["unknown"])[0]
        print(
            f"  [{sample_idx}/{args.max_samples or 'all'}] {batch['scenario_name'][0]} | "
            f"type={sample_type} K={used_k}/{original_k} N={target_frames} -> {sample_dir}"
        )

    elapsed = time.time() - start_time
    print(f"\nInference complete. {sample_idx} samples in {elapsed:.1f}s ({elapsed/max(sample_idx,1):.1f}s/sample)")
    print(f"Output: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.1 SOTA Video-to-Video Inference - AI City 2026 Track 5"
    )
    parser.add_argument("--manifest", type=str,
                        default="/workspace/wts_baseline/data/manifests/test.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/wts_baseline/prediction_wan_sota")
    parser.add_argument("--max_samples", type=int, default=3)

    # WAN model
    parser.add_argument("--model_id", type=str,
                        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")

    # Resolution
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--output_width", type=int, default=1280)
    parser.add_argument("--output_height", type=int, default=720)

    # Sampling
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--min_guidance", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=81)
    parser.add_argument("--overlap_frames", type=int, default=2)

    # History
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--max_history_frames", type=int, default=None)
    parser.add_argument("--history_strategy", choices=["tail", "dense_tail", "uniform"],
                        default="dense_tail")

    # SOTA features
    parser.add_argument("--use_background_preservation", action="store_true", default=True)
    parser.add_argument("--use_alpha_compositing", action="store_true", default=True)
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=True)
    parser.add_argument("--use_step_caching", action="store_true", default=True)

    # Data
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    # Map manifest to /data if it points there
    if not os.path.exists(args.manifest):
        alt = os.path.join("/data/Track_5/WTS_dataset/WTS_dataset", args.manifest)
        if os.path.exists(alt):
            args.manifest = alt

    infer(args)


if __name__ == "__main__":
    main()
