import json
import os

def main():
    manifest_path = "/workspace/wts_baseline/data/manifests/test_sota_compiled.jsonl"
    if not os.path.exists(manifest_path):
        print("Manifest not found.")
        return
        
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            scenario = item["scenario_name"]
            camera = item.get("vlm_analysis", {}).get("camera", {})
            vtype = camera.get("view_type", "")
            motion = camera.get("motion", "")
            print(f"{scenario} | view_type: '{vtype}' | motion: '{motion}'")

if __name__ == "__main__":
    main()
