import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    print("Loading Qwen2-VL-2B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    image_path = "/data/Track_5/WTS_dataset/WTS_dataset/WTS_TRACK5_TEST/WTS_TRACK5_TEST/video2189/input/84.png"
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)

    # Prepare conversation prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the static background, road layout, road markings, and buildings in this driving video frame. Focus on details like arches, bridge, vehicle models, or text on the road."}
            ]
        }
    ]

    # Preparation for inference
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

    print("Generating description...")
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("\nGenerated description:")
    print("-" * 50)
    print(output_text[0])
    print("-" * 50)

if __name__ == "__main__":
    main()
