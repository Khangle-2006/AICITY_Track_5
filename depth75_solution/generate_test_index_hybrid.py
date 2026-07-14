import json
import csv
import os

def main():
    manifest_path = "/workspace/wts_baseline/data/manifests/test_sota_compiled.jsonl"
    csv_path = "/workspace/depth75_solution/test_index.csv"
    out_csv_path = "/workspace/depth75_solution/test_index_hybrid.csv"

    if not os.path.exists(csv_path):
        print(f"Error: Base CSV {csv_path} does not exist!")
        return

    # Load view types from manifest
    sample_to_view = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                scenario = item["scenario_name"]
                camera = item.get("vlm_analysis", {}).get("camera", {})
                vtype = camera.get("view_type", "").lower()
                
                if "overhead" in vtype or "fixed" in vtype or "security" in vtype:
                    sample_to_view[scenario] = "wts_overhead_or_fixed_camera"
                else:
                    sample_to_view[scenario] = "wts_vehicle_or_ip_camera"
    else:
        print(f"Warning: Manifest {manifest_path} not found. Using fallback classification.")

    # Read test_index.csv and update view_type
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in sample_to_view:
                row["view_type"] = sample_to_view[sample_id]
            else:
                # Fallback heuristics
                if sample_id.startswith("video"):
                    row["view_type"] = "wts_vehicle_or_ip_camera"
                else:
                    row["view_type"] = "wts_overhead_or_fixed_camera"
            rows.append(row)

    # Write test_index_hybrid.csv
    with open(out_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    counts = {}
    for r in rows:
        counts[r["view_type"]] = counts.get(r["view_type"], 0) + 1
    print("Classification summary:", counts)

if __name__ == "__main__":
    main()
