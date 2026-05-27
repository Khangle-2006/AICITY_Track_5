"""
Subprocess: Load UMT5 text_encoder, encode all prompts from manifest, save to .pt files, exit.
This guarantees clean GPU memory for the main inference process.
"""

import argparse
import json
import os
import sys
import torch
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.wts_dataset import WTSDataset
from src.pipelines.wan_video_pipeline import build_wan_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--output_dir", type=str, default="/tmp/prompt_embeds")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--history_strategy", type=str, default="dense_tail")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load dataset to get prompts
    dataset = WTSDataset(
        args.manifest,
        is_train=False,
        size=(args.height, args.width),
        max_history_frames=args.h,
        history_strategy=args.history_strategy,
    )
    if args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]

    print(f"Loading text_encoder for {args.model_id} ...")
    from diffusers import WanVideoToVideoPipeline as HFP
    pipe = HFP.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe.text_encoder.to(device)
    print(f"  GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, sample in enumerate(dataset.samples):
        caption = sample.get("merged_caption", sample.get("text", ""))
        scenario_name = sample.get("scenario_name", f"sample_{idx}")
        phase_label = torch.tensor(sample.get("phase_label", 0), device=device)
        sample_type = sample.get("sample_type", "unknown")
        prompt = build_wan_prompt(caption, phase_label, sample_type)

        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=args.max_sequence_length,
            device=device,
        )

        out_path = os.path.join(args.output_dir, f"{scenario_name}.pt")
        torch.save(
            {
                "prompt_embeds": prompt_embeds.cpu(),
                "negative_prompt_embeds": negative_prompt_embeds.cpu(),
            },
            out_path,
        )
        print(f"  [{idx+1}/{len(dataset.samples)}] {scenario_name} -> {out_path}")

    print(f"All prompts encoded. GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
