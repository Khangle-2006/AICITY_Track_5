"""
Compare WAN vs SOTA on first 10 validation samples.
Pre-encode prompts on CPU to avoid OOM.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5Tokenizer

from src.datasets.collate import wts_collate_fn
from src.datasets.wts_dataset import WTSDataset
from src.datasets.transforms import denormalize_image
from src.pipelines.wan_video_pipeline import (
    WanPipelineConfig, WanSOTAPipeline, build_wan_prompt,
)

import torchvision


MANIFEST = "/workspace/wts_baseline/data/manifests/val_sota.jsonl"
OUTPUT_DIR = "/tmp/wan_val_pred"
MAX_SAMPLES = 10
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
torch_dtype = torch.bfloat16

dataset = WTSDataset(
    MANIFEST, is_train=False, size=(480, 832),
    max_history_frames=16, history_strategy="dense_tail",
)
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False,
    collate_fn=wts_collate_fn, num_workers=2,
)

print("Pre-loading prompts...")
prompts = []
gt_frames_list = []
batch_list = []
for i, batch in enumerate(dataloader):
    if i >= MAX_SAMPLES:
        break
    batch_list.append(batch)
    phase_label = batch.get("phase_label", torch.zeros(1, dtype=torch.long))[0]
    sample_type = batch.get("sample_type", ["unknown"])[0]
    caption = batch.get("merged_caption", [""])[0]
    prompt = build_wan_prompt(caption, phase_label, sample_type)
    prompts.append(prompt)

print(f"Loaded {len(prompts)} samples. Pre-encoding prompts on CPU...")

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder", torch_dtype=torch_dtype,
)
text_encoder.eval()

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "deformed, disfigured, misshapen, still picture, messy background"
)

def encode_on_cpu(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeds = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state
    return embeds.to(torch_dtype)

prompt_embeds_list = [encode_on_cpu(p) for p in prompts]
neg_prompt_embeds = encode_on_cpu(NEGATIVE_PROMPT)

del text_encoder
del tokenizer
print("Prompt encoding done.")

print("Loading WAN pipeline...")
config = WanPipelineConfig(
    model_id=MODEL_ID,
    torch_dtype="bfloat16", height=480, width=832,
    num_inference_steps=50, guidance_scale=5.0, min_guidance=1.0,
    strength=0.7, flow_shift=3.0, chunk_size=81, overlap_frames=2,
    output_width=1280, output_height=720,
    use_background_preservation=True, use_alpha_compositing=True,
    use_dynamic_cfg=True, use_step_caching=True,
)
pipeline = WanSOTAPipeline(config)
print("Pipeline loaded.")

sample_idx = 0
start_time = time.time()

for batch, prompt_embeds in zip(batch_list, prompt_embeds_list):
    scenario = batch["scenario_name"][0]
    num_future = batch["future_frames"].size(1)
    print(f"\n[{sample_idx + 1}/{MAX_SAMPLES}] {scenario} generating {num_future} frames ...")
    gen_start = time.time()

    batch["frame_length"] = num_future
    pred_frames = pipeline.generate_batch(
        batch,
        prompt_embeds=prompt_embeds.to(device),
        negative_prompt_embeds=neg_prompt_embeds.to(device),
    )

    pred = denormalize_image(pred_frames[0])
    pred_resized = pipeline.resize_for_submission(pred)

    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(pred_resized.size(0)):
        torchvision.utils.save_image(pred_resized[i], os.path.join(sample_dir, f"{i:04d}.png"))

    if "future_frames" in batch:
        gt = denormalize_image(batch["future_frames"])[0]
        gt_resized = pipeline.resize_for_submission(gt)
        gt_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_gt")
        os.makedirs(gt_dir, exist_ok=True)
        for i in range(gt_resized.size(0)):
            torchvision.utils.save_image(gt_resized[i], os.path.join(gt_dir, f"{i:04d}.png"))

    elapsed = time.time() - gen_start
    print(f"  Done in {elapsed:.1f}s -> sample_{sample_idx}")
    sample_idx += 1

total_time = time.time() - start_time
print(f"\nWAN inference complete: {sample_idx} samples in {total_time:.1f}s")

print("\nRunning evaluation...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from eval_sota import evaluate_metrics
evaluate_metrics(OUTPUT_DIR, manifest_path=MANIFEST, use_lpips=True, use_clip=True, use_fvd=True)
