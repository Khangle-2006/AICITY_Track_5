import json
import os

# Original details from test_video2189.jsonl
merged_caption = "The vehicle is decelerating as it travels behind and to the left of the pedestrian. It is a far distance away from the pedestrian. The pedestrian is visible within the vehicle's field of view. The vehicle is currently going straight ahead at a speed of 20km/h. The vehicle is in an urban environment, on a clear and bright day. The road conditions are dry, the road is level, and the surface is made of asphalt. The traffic volume is typical, and the road is classified as a main road with one-way traffic and three lanes. There are sidewalks on both sides of the road, and guardrails are present. Street lights also illuminate the surroundings. These are the only details known about the event involving the vehicle and its surroundings. A male pedestrian in his 20s, measuring approximately 180 cm in height, is wearing navy blue upper body clothing and black short pants. He is located diagonally to the right, in front of a vehicle on a main road in an urban setting. The pedestrian's body orientation aligns with the same direction as the vehicle, and his line of sight is fixed towards the front, in the direction of travel. Unaware of the vehicle's presence, he is crossing the road immediately in front of or behind the moving vehicle at a high speed. The weather is clear, with bright lighting conditions, and the road surface is dry and level. Both sides of the road have sidewalks, and there are guardrails and street lights present along the way. The event occurs on a weekday with usual traffic volume."
phase_label = 3
scenario_name = "video2189"
frame_length = 87

base_path = "/data/Track_5/WTS_dataset/WTS_dataset/WTS_TRACK5_TEST/WTS_TRACK5_TEST/video2189/input"

# Stride 2 (spacing of 1 frame): e.g., 84, 82, 80, ..., 54 (16 frames)
stride2_indices = list(range(84, 53, -2))[::-1]
stride2_frames = [os.path.join(base_path, f"{i}.png") for i in stride2_indices]

# Stride 3 (spacing of 2 frames): e.g., 84, 81, 78, ..., 39 (16 frames)
stride3_indices = list(range(84, 38, -3))[::-1]
stride3_frames = [os.path.join(base_path, f"{i}.png") for i in stride3_indices]

manifest_stride2 = {
    "history_frames": stride2_frames,
    "merged_caption": merged_caption,
    "phase_label": phase_label,
    "scenario_name": scenario_name,
    "frame_length": frame_length
}

manifest_stride3 = {
    "history_frames": stride3_frames,
    "merged_caption": merged_caption,
    "phase_label": phase_label,
    "scenario_name": scenario_name,
    "frame_length": frame_length
}

os.makedirs("/workspace/wts_baseline/data/manifests", exist_ok=True)

with open("/workspace/wts_baseline/data/manifests/test_video2189_stride2.jsonl", "w") as f:
    f.write(json.dumps(manifest_stride2) + "\n")

with open("/workspace/wts_baseline/data/manifests/test_video2189_stride3.jsonl", "w") as f:
    f.write(json.dumps(manifest_stride3) + "\n")

print("Created manifests successfully:")
print(f"Stride 2 history indices: {stride2_indices} (length {len(stride2_frames)})")
print(f"Stride 3 history indices: {stride3_indices} (length {len(stride3_frames)})")
