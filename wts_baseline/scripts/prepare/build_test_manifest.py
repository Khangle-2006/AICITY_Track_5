import os
import json
import argparse
from pathlib import Path

def build_test_manifest(test_dir, output_path, h=16):
    test_dir = Path(test_dir)
    scenarios = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    valid_samples = []
    
    print(f"Found {len(scenarios)} test scenarios in {test_dir}")
    
    for scenario_dir in scenarios:
        scenario_name = scenario_dir.name
        
        # Load caption.json
        caption_path = scenario_dir / "caption.json"
        if not caption_path.exists():
            print(f"[WARN] No caption.json in {scenario_name}, skipping.")
            continue
            
        with open(caption_path, 'r') as f:
            caption_data = json.load(f)
            
        event_phase = caption_data.get("event_phase", [])
        if not event_phase:
            print(f"[WARN] Empty event_phase in {scenario_name}, skipping.")
            continue
            
        phase = event_phase[0]
        cap_ped = phase.get("caption_pedestrian", "")
        cap_veh = phase.get("caption_vehicle", "")
        labels = phase.get("labels", [])
        
        merged_caption = f"{cap_veh} {cap_ped}".strip()
        phase_label = int(labels[0]) if labels else 0
        
        # Find input frames
        input_dir = scenario_dir / "input"
        if not input_dir.exists():
            print(f"[WARN] No input dir in {scenario_name}, skipping.")
            continue
            
        # Get list of png frames and sort numerically
        png_files = sorted(list(input_dir.glob("*.png")), key=lambda x: int(x.stem))
        if not png_files:
            print(f"[WARN] No png frames in {scenario_name}/input, skipping.")
            continue
            
        # Select/construct exactly H history frames
        F = len(png_files)
        history_paths = []
        if F >= h:
            # Take the last H frames
            selected_files = png_files[-h:]
            history_paths = [str(p) for p in selected_files]
        else:
            # Pad by repeating the first frame
            pad_count = h - F
            selected_files = [png_files[0]] * pad_count + png_files
            history_paths = [str(p) for p in selected_files]
            
        valid_samples.append({
            "history_frames": history_paths,
            "merged_caption": merged_caption,
            "phase_label": phase_label,
            "scenario_name": scenario_name,
            "frame_length": caption_data.get("frame length", 8)
        })
        
    # Save manifest
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for s in valid_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Successfully processed {len(valid_samples)} samples.")
    print(f"Manifest saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str,
                        default="/data/Track_5/WTS_dataset/WTS_dataset/WTS_TRACK5_TEST/WTS_TRACK5_TEST")
    parser.add_argument("--output", type=str,
                        default="/workspace/wts_baseline/data/manifests/test.jsonl")
    parser.add_argument("--h", type=int, default=16)
    args = parser.parse_args()
    
    build_test_manifest(args.test_dir, args.output, h=args.h)

if __name__ == "__main__":
    main()
