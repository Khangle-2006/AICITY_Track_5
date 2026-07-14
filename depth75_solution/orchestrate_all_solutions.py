import os
import subprocess
import time
import sys
import shutil

def run_command(cmd, env_updates=None):
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    print(f"\nRunning command: {' '.join(cmd)}")
    sys.stdout.flush()
    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            sys.exit(process.returncode)
        else:
            print("Command completed successfully!")
            sys.stdout.flush()
    except Exception as e:
        print(f"Exception during command execution: {e}")
        sys.exit(1)

def main():
    print("=== STARTING FULL SOLUTIONS ORCHESTRATOR ===")
    sys.stdout.flush()

    # Step 1: Extract prediction_sft_2000.tar.gz
    archive_path = "/workspace/prediction_sft_2000.tar.gz"
    dest_dir = "/workspace/Cosmo/prediction_epoch3"
    
    if not os.path.exists(archive_path):
        print(f"Error: Archive {archive_path} not found on the server!")
        sys.exit(1)
        
    if not os.path.exists(os.path.join(dest_dir, "video1170", "0.png")):
        print(f"\n--- EXTRACTING SFT PREDICTIONS ARCHIVE TO {dest_dir} ---")
        os.makedirs(dest_dir, exist_ok=True)
        # Extract and strip the parent directory "prediction_sft_2000"
        cmd_extract = ["tar", "-xf", archive_path, "-C", dest_dir, "--strip-components=1"]
        run_command(cmd_extract)
    else:
        print("Raw SFT predictions already extracted.")

    # Step 1.5: Generate Classical Baseline predictions
    baseline_manifest = "/workspace/depth75_solution/wts_classical_baseline/manifest.csv"
    if os.path.exists(baseline_manifest) and os.path.getsize(baseline_manifest) > 100:
        print("Classical baseline predictions already generated.")
    else:
        print("\n--- GENERATING CLASSICAL BASELINE PREDICTIONS ---")
        sys.stdout.flush()
        cmd_baseline = [
            "/opt/conda/bin/python", "-m", "baseline.cli",
            "--index_dir", "/workspace/depth75_solution",
            "--out", "/workspace/depth75_solution/wts_classical_baseline",
            "--method", "actor_layer_small_filled",
            "--clean"
        ]
        run_command(cmd_baseline, env_updates={"PYTHONPATH": "/workspace/depth75_solution"})

    # Step 2: Run Batch Swin2SR on all 71 samples
    print("\n--- RUNNING BATCH SWIN2SR UPSCALING ON ALL 71 SAMPLES ---")
    sys.stdout.flush()
    cmd_sr = [
        "/opt/conda/bin/python", "/workspace/depth75_solution/upscale_predictions_sr.py",
        "--input_dir", "/workspace/Cosmo/prediction_epoch3",
        "--output_dir", "/workspace/Cosmo/prediction_epoch3_sr",
        "--test_csv", "/workspace/depth75_solution/test_index_hybrid.csv",
        "--device", "cuda:2",
        "--batch_size", "4"
    ]
    run_command(cmd_sr)

    # Setup temporary index directory for build_hybrid_submission
    temp_index_dir = "/workspace/depth75_solution/temp_index_dir"
    os.makedirs(temp_index_dir, exist_ok=True)
    shutil.copy2("/workspace/depth75_solution/test_index_hybrid.csv", os.path.join(temp_index_dir, "test_index.csv"))
    print(f"Created temporary index directory: {temp_index_dir}")

    try:
        # Step 3: Build Hybrid Version A (Raw Cosmo + SR + Classical Overhead)
        print("\n--- BUILDING HYBRID VERSION A (RAW COSMO + SR + CLASSICAL OVERHEAD) ---")
        sys.stdout.flush()
        cmd_hybrid_a = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.build_hybrid_submission",
            "--index_dir", temp_index_dir,
            "--wan_dir", "/workspace/Cosmo/prediction_epoch3_sr",
            "--region_dir", "/workspace/depth75_solution/wts_classical_baseline",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_raw",
            "--report_csv", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_raw_report.csv",
            "--wan_label", "cosmo_sft_sr",
            "--region_label", "actor_layer_small_filled",
            "--region_views", "wts_overhead_or_fixed_camera",
            "--zip", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_raw.zip",
            "--clean"
        ]
        run_command(cmd_hybrid_a, env_updates={"PYTHONPATH": "/workspace/depth75_solution"})

        # Step 4: Build Pure Version A (Pure Raw Cosmo + SR for all 71 samples)
        print("\n--- BUILDING PURE VERSION A (PURE RAW COSMO + SR FOR ALL 71 SAMPLES) ---")
        sys.stdout.flush()
        cmd_pure_a = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.build_hybrid_submission",
            "--index_dir", temp_index_dir,
            "--wan_dir", "/workspace/Cosmo/prediction_epoch3_sr",
            "--region_dir", "/workspace/Cosmo/prediction_epoch3_sr",  # use same directory so all views are raw cosmo + sr
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_pure_sr_raw",
            "--report_csv", "/workspace/depth75_solution/cosmos_epoch3_pure_sr_raw_report.csv",
            "--wan_label", "cosmo_sft_sr",
            "--region_label", "cosmo_sft_sr",
            "--region_views", "wts_overhead_or_fixed_camera",
            "--zip", "/workspace/depth75_solution/cosmos_epoch3_pure_sr_raw.zip",
            "--clean"
        ]
        run_command(cmd_pure_a, env_updates={"PYTHONPATH": "/workspace/depth75_solution"})

        # Step 5: Run Post-Refinement Renderer (Baseline Configuration)
        print("\n--- RUNNING POST-REFINEMENT RENDERER (BASELINE CONFIGURATION) ---")
        sys.stdout.flush()
        cmd_refine_base = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.materialize_depth_layered_renderer_test",
            "--test_csv", "/workspace/depth75_solution/test_index_hybrid.csv",
            "--base_dir", "/workspace/Cosmo/prediction_epoch3_sr",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined",
            "--model_cache", "/workspace/depth75_solution/model_cache",
            "--view_type", "wts_vehicle_or_ip_camera",
            "--alpha", "0.75",
            "--focal_scale", "0.9",
            "--point_stride", "2",
            "--splat_radius", "1",
            "--min_matches", "20",
            "--min_inliers", "12",
            "--photo_sigma", "32",
            "--segment_batch", "4",
            "--pose_smoothing_window", "7",
            "--filled_pose_scale", "0.35",
            "--confidence_ema", "0.65",
            "--matcher", "orb",
            "--depth_model", "midas",
            "--segment_model", "deeplabv3",
            "--clean"
        ]
        run_command(cmd_refine_base, env_updates={
            "PYTHONPATH": "/workspace/depth75_solution",
            "CUDA_VISIBLE_DEVICES": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        })

        # Step 6: Build Hybrid Version B (Baseline Refined Cosmo + SR + Classical Overhead)
        print("\n--- BUILDING HYBRID VERSION B (BASELINE REFINED COSMO + SR + CLASSICAL OVERHEAD) ---")
        sys.stdout.flush()
        cmd_hybrid_b = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.build_hybrid_submission",
            "--index_dir", temp_index_dir,
            "--wan_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined",
            "--region_dir", "/workspace/depth75_solution/wts_classical_baseline",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_refined",
            "--report_csv", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_refined_report.csv",
            "--wan_label", "depth_layered_sr",
            "--region_label", "actor_layer_small_filled",
            "--region_views", "wts_overhead_or_fixed_camera",
            "--zip", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_refined.zip",
            "--clean"
        ]
        run_command(cmd_hybrid_b, env_updates={"PYTHONPATH": "/workspace/depth75_solution"})

        # Step 7: Run Post-Refinement Renderer (SOTA Configuration: SuperPoint + DepthAnything + YOLOv8 + RAFT + Depth-weighting)
        print("\n--- RUNNING POST-REFINEMENT RENDERER (SOTA CONFIGURATION FOR VEHICLE VIEWS) ---")
        sys.stdout.flush()
        cmd_refine_sota = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.materialize_depth_layered_renderer_test",
            "--test_csv", "/workspace/depth75_solution/test_index_hybrid.csv",
            "--base_dir", "/workspace/Cosmo/prediction_epoch3_sr",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined_sota",
            "--model_cache", "/workspace/depth75_solution/model_cache",
            "--view_type", "wts_vehicle_or_ip_camera",
            "--alpha", "0.75",
            "--focal_scale", "0.9",
            "--point_stride", "2",
            "--splat_radius", "1",
            "--min_matches", "20",
            "--min_inliers", "12",
            "--photo_sigma", "32",
            "--segment_batch", "4",
            "--pose_smoothing_window", "7",
            "--filled_pose_scale", "0.35",
            "--confidence_ema", "0.65",
            "--matcher", "superpoint",
            "--depth_model", "depthanything",
            "--segment_model", "yolov8",
            "--disocclusion_inpainting",
            "--alpha_depth_weighting",
            "--clean"
        ]
        run_command(cmd_refine_sota, env_updates={
            "PYTHONPATH": "/workspace/depth75_solution",
            "CUDA_VISIBLE_DEVICES": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        })

        # Step 8: Build Hybrid Version C (SOTA Refined Cosmo + SR + Classical Overhead)
        print("\n--- BUILDING HYBRID VERSION C (SOTA REFINED COSMO + SR + CLASSICAL OVERHEAD) ---")
        sys.stdout.flush()
        cmd_hybrid_c = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.build_hybrid_submission",
            "--index_dir", temp_index_dir,
            "--wan_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined_sota",
            "--region_dir", "/workspace/depth75_solution/wts_classical_baseline",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_refined_sota",
            "--report_csv", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_refined_sota_report.csv",
            "--wan_label", "depth_layered_sr_sota",
            "--region_label", "actor_layer_small_filled",
            "--region_views", "wts_overhead_or_fixed_camera",
            "--zip", "/workspace/depth75_solution/cosmos_epoch3_hybrid_sr_refined_sota.zip",
            "--clean"
        ]
        run_command(cmd_hybrid_c, env_updates={"PYTHONPATH": "/workspace/depth75_solution"})

        # Step 9: Run Post-Refinement Renderer (SOTA Configuration for ALL 71 views)
        print("\n--- RUNNING POST-REFINEMENT RENDERER (SOTA CONFIGURATION FOR ALL 71 VIEWS) ---")
        sys.stdout.flush()
        
        # We need a new hybrid test CSV where view_type is "test" for all, so that the script processes all 71
        # The original test_index.csv is perfect for this!
        cmd_refine_sota_all = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.materialize_depth_layered_renderer_test",
            "--test_csv", "/workspace/depth75_solution/test_index.csv",
            "--base_dir", "/workspace/Cosmo/prediction_epoch3_sr",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined_sota_all",
            "--model_cache", "/workspace/depth75_solution/model_cache",
            "--view_type", "test",
            "--alpha", "0.75",
            "--focal_scale", "0.9",
            "--point_stride", "2",
            "--splat_radius", "1",
            "--min_matches", "20",
            "--min_inliers", "12",
            "--photo_sigma", "32",
            "--segment_batch", "4",
            "--pose_smoothing_window", "7",
            "--filled_pose_scale", "0.35",
            "--confidence_ema", "0.65",
            "--matcher", "superpoint",
            "--depth_model", "depthanything",
            "--segment_model", "yolov8",
            "--disocclusion_inpainting",
            "--alpha_depth_weighting",
            "--clean"
        ]
        run_command(cmd_refine_sota_all, env_updates={
            "PYTHONPATH": "/workspace/depth75_solution",
            "CUDA_VISIBLE_DEVICES": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        })

        # Step 10: Build Pure Version B (Pure Refined SOTA Cosmo + SR for all 71 samples)
        print("\n--- BUILDING PURE VERSION B (PURE REFINED SOTA COSMO + SR FOR ALL 71 SAMPLES) ---")
        sys.stdout.flush()
        cmd_pure_b = [
            "/opt/conda/bin/python", "-m", "wan_ticft.evaluation.build_hybrid_submission",
            "--index_dir", temp_index_dir,
            "--wan_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined_sota_all",
            "--region_dir", "/workspace/depth75_solution/cosmos_epoch3_sr_refined_sota_all",
            "--out_dir", "/workspace/depth75_solution/cosmos_epoch3_pure_sr_refined_sota",
            "--report_csv", "/workspace/depth75_solution/cosmos_epoch3_pure_sr_refined_sota_report.csv",
            "--wan_label", "depth_layered_sr_sota",
            "--region_label", "depth_layered_sr_sota",
            "--region_views", "wts_overhead_or_fixed_camera",
            "--zip", "/workspace/depth75_solution/cosmos_epoch3_pure_sr_refined_sota.zip",
            "--clean"
        ]
        run_command(cmd_pure_b, env_updates={"PYTHONPATH": "/workspace/depth75_solution"})

    finally:
        # Clean up temporary index directory
        if os.path.exists(temp_index_dir):
            shutil.rmtree(temp_index_dir)
            print(f"Cleaned up temporary index directory: {temp_index_dir}")

    print("\n=== ALL SOLUTIONS ORCHESTRATION COMPLETE! ===")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
