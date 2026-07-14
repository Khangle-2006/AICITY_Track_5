import torch
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    # 1. Load history frames (16 frames)
    history_dir = "/data/Track_5/WTS_dataset/WTS_dataset/WTS_TRACK5_TEST/WTS_TRACK5_TEST/video2189/input"
    frame_indices = range(69, 85) # 16 frames: 69 to 84
    
    print(f"Loading {len(frame_indices)} history frames from {history_dir}...")
    frames = []
    for idx in frame_indices:
        path = os.path.join(history_dir, f"{idx}.png")
        if os.path.exists(path):
            frames.append(Image.open(path))
        else:
            print(f"Warning: {path} not found.")
            
    if not frames:
        print("No frames loaded. Exiting.")
        return

    print("Loading Qwen2-VL-2B-Instruct on GPU 2...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # 2. Build messages with video frames
    # Qwen2-VL processor accepts a list of PIL Images as a video input
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "fps": 5.0 # Set temporal rate
                },
                {
                    "type": "text", 
                    "text": "Describe the camera motion (e.g. translation direction, forward speed), static road layout (concrete arches, markings), and actions of pedestrians or vehicles in this driving video clip."
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda:2")

    print("Generating motion description using Video VLM...")
    generated_ids = model.generate(**inputs, max_new_tokens=150)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*50)
    print("Video VLM Generated Motion Description:")
    print("-" * 50)
    print(output_text[0])
    print("="*50)

if __name__ == "__main__":
    main()
