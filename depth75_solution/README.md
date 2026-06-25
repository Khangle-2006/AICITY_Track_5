# AI City Challenge Track 5 - Depth-Layered A75 Solution

This folder contains the code used for the `depth_layered_a75` submission route for
AI City Challenge Track 5 video prediction.

The method is intentionally inference-only. It starts from an already generated
BDD future-video prediction, then improves static road/background alignment with
a depth-aware 3D forward-splat from the last observed frame. The WTS side uses a
classical optical-flow baseline with an object-layer route.

## Public Candidate

Best tracked public candidate from this route:

```text
hybrid_depth_layered_a75_wts_overhead_object_layer_candidate
Score 69.2456
PSNR 18.5942
SSIM 0.6222
LPIPS 0.3312
CLIP 0.9397
FID 47.8320
FVD 31.2770
```

## What Is Included

```text
depth75_solution/
  baseline/                         # Small baseline utilities and WTS generators
  wan_ticft/
    evaluation/
      materialize_depth_layered_renderer.py
      materialize_depth_layered_renderer_test.py
      build_hybrid_submission.py
      build_depth_layered_a75_wts_overhead_candidate.sh
      evaluate_video_reports.py
      check_gpu_free.sh
  scripts/
    build_depth75_candidate.sh       # Recommended entrypoint
  requirements.txt
```

Large files are intentionally not included:

- Track 5 dataset
- generated predictions
- model caches/checkpoints
- submission folders/zips
- third-party model weights

## Method Summary

For BDD front-camera test videos:

1. Use the existing BDD base prediction directory, typically
   `cnn_ltxmix65_bdd_test`.
2. For each sample, take the last observed frame as a sharp texture/depth anchor.
3. Estimate relative depth with `Intel/dpt-hybrid-midas`.
4. Segment dynamic actor regions with DeepLabV3.
5. Estimate camera pose from the anchor to each generated future frame using
   ORB matches and PnP.
6. Forward-splat static anchor pixels into each future frame with z-buffering.
7. Blend only confident static/background pixels into the base prediction.
8. Keep actors, unreliable regions, and disocclusions from the base prediction.

For WTS:

1. Generate overhead/fixed-camera outputs with `actor_layer_small_filled`.
2. Generate vehicle/IP-camera outputs with `region_norm_medium`.
3. Build one hybrid submission folder: BDD from depth-layered output, WTS from
   the WTS route.

## Expected Runtime Layout

The scripts assume this common workspace layout:

```text
/workspace/
  depth75_solution/                  # This folder
  experiments/
    index/
      test_index.csv
    wan_ticft/
      cnn_ltxmix65_bdd_test/         # Required BDD base predictions
      wts_overhead_object_layer_vehicle_medium_test/  # Optional; generated if missing
```

If your folder is inside the cloned GitHub repo, either run from there or copy
`depth75_solution` to `/workspace/depth75_solution`.

## Environment

The original environment used Conda env `wan22`.

Install Python packages if needed:

```bash
pip install -r depth75_solution/requirements.txt
```

The first depth render downloads/caches:

- `Intel/dpt-hybrid-midas`
- TorchVision DeepLabV3 ResNet50 weights

GPU is required for the depth/segmentation pass. The WTS classical generators
run on CPU.

## Build The Depth75 Candidate

From `/workspace`:

```bash
cd /workspace
conda activate wan22

CUDA_VISIBLE_DEVICES=2 bash depth75_solution/scripts/build_depth75_candidate.sh
```

Default outputs:

```text
/workspace/experiments/wan_ticft/depth_layered_a75_bdd_test
/workspace/experiments/wan_ticft/wts_overhead_object_layer_vehicle_medium_test
/workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate
/workspace/experiments/wan_ticft/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate_report.csv
```

The script verifies the final folder but does not create a zip by default.

## Important Inputs

`cnn_ltxmix65_bdd_test` must already exist. It is the BDD base future prediction
directory that the depth-layered renderer refines. It must contain one folder per
BDD test `source_folder`, with numbered PNG frames:

```text
/workspace/experiments/wan_ticft/cnn_ltxmix65_bdd_test/<source_folder>/0.png
/workspace/experiments/wan_ticft/cnn_ltxmix65_bdd_test/<source_folder>/1.png
...
```

To override paths:

```bash
BDD_BASE=/path/to/bdd_base \
WTS=/path/to/wts_outputs \
SUBMISSION=/path/to/final_candidate \
CUDA_VISIBLE_DEVICES=2 \
bash depth75_solution/scripts/build_depth75_candidate.sh
```

If `WTS` does not exist, the wrapper generates it from `test_index.csv`.

## Create A Zip For Submission

After verifying the output folder:

```bash
cd /workspace
python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir /workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate \
  --submission_dir /workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate \
  --zip /workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate.zip
```

If running from this standalone folder, include it on `PYTHONPATH`:

```bash
PYTHONPATH=/workspace/depth75_solution python -m baseline.submission ...
```

## Validate Existing Output

```bash
PYTHONPATH=/workspace/depth75_solution python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir /workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate \
  --submission_dir /workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate \
  --verify_only
```

## Notes

- `alpha=0.75` is the validated depth blend strength for this route.
- The renderer avoids actor regions; it is mainly a static background/road
  geometry improvement.
- Do not commit generated folders, model caches, datasets, or zip files to Git.
