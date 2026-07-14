import torch
import os
import json
import argparse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/manifests/test_video2189_stride2.jsonl")
    parser.add_argument("--output_file", type=str, default="wts_baseline/prediction_wan2.2_ticft_sr/sa2va_understanding.json")
    args = parser.parse_args()

    # Load manifest
    print(f"Reading manifest from {args.manifest}...")
    samples = []
    with open(args.manifest, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    print(f"Loaded {len(samples)} samples to process.")

    # Initialize model
    print("Loading Qwen2-VL-2B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    understandings = {}
    
    # Load existing if available
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r") as f:
                understandings = json.load(f)
            print(f"Loaded {len(understandings)} existing understandings.")
        except Exception as e:
            print(f"Could not load existing file: {e}")

    for idx, sample in enumerate(samples):
        scenario_name = sample.get("scenario_name")
        if scenario_name in understandings:
            print(f"[{idx+1}/{len(samples)}] Skipping {scenario_name} (already exists)")
            continue

        history_paths = sample.get("history_frames", [])
        if not history_paths:
            print(f"[{idx+1}/{len(samples)}] Warning: no history frames for {scenario_name}")
            continue

        # Load frames
        frames = []
        for path in history_paths:
            if not os.path.exists(path):
                # Fallback to absolute/relative paths check
                if path.startswith("/"):
                    alt_path = path
                else:
                    alt_path = os.path.join("/workspace/wts_baseline", path)
                
                if not os.path.exists(alt_path):
                    print(f"Warning: frame path {path} not found.")
                    continue
                path = alt_path
            frames.append(Image.open(path))

        if not frames:
            print(f"[{idx+1}/{len(samples)}] No valid frames for {scenario_name}")
            continue

        print(f"[{idx+1}/{len(samples)}] Processing {scenario_name} (loaded {len(frames)} frames)...")
        
        # Build prompt for video understanding
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "fps": 5.0
                    },
                    {
                        "type": "text",
                        "text": "Describe the camera motion (e.g. forward translation speed, parallax), static road layout (concrete structures, lanes, asphalt markings), and key landmarks in this driving video clip."
                    }
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda:2")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=150)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        description = output_text[0].strip()
        print(f"Description: {description}")

        understandings[scenario_name] = {
            "sa2va_description": description
        }

        # Save checkpoint
        with open(args.output_file, "w") as f:
            json.dump(understandings, f, indent=2)

    print(f"Completed! Saved VLM understandings to {args.output_file}")

if __name__ == "__main__":
    main()
