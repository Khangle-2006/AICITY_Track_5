# AI City Challenge Track 5 - Multiframe LTXHF-T55 Solution

This folder contains the reproducibility package for the tracked
`hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate` route used during AI City
Challenge Track 5 future-frame prediction experiments.

The route is inference-only after the BDD base predictions are available. It
combines:

- BDD front-camera predictions from the LTXHF-T55 base route.
- A multiframe depth-layered temporal renderer that reinforces static geometry
  from observed history frames.
- WTS predictions from the classical overhead/object-layer and vehicle/IP
  camera routes.
- A deterministic hybrid assembler and submission verifier.

## Candidate Reference

Best public score tracked for this exact route:

```text
candidate  hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate
split      Public
score      69.3815
```

The reference build report is committed at:

```text
multiframe_ltxhf_t55_solution/reports/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv
```

Expected submission shape:

```text
samples  71
frames   5194
```

## What Is Included

```text
multiframe_ltxhf_t55_solution/
  baseline/
    cli.py                              # WTS generators and utility entrypoints
    frame_io.py
    index_io.py
    submission.py                       # Track 5 folder verifier and zipper
    verify.py
    ...
  wan_ticft/
    evaluation/
      build_hybrid_submission.py
      materialize_depth_layered_renderer.py
      materialize_depth_layered_renderer_test.py
      materialize_depth_layered_multiframe.py
      materialize_depth_layered_multiframe_test.py
      run_ltxv_test_bdd.py              # Optional BDD base-generation helper
      ltxv_2b_gate.yaml                 # Optional LTX-Video helper config
      evaluate_video_reports.py
      check_gpu_free.sh
  scripts/
    build_multiframe_ltxhf_t55_candidate.sh
    create_submission_zip.sh
    validate_candidate_report.py
  reports/
    hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv
  requirements.txt
```

Large artifacts are intentionally not committed:

- Track 5 train/test data.
- Generated PNG prediction folders.
- Submission zip files.
- Model checkpoints and Hugging Face caches.
- The precomputed LTXHF-T55 BDD base prediction folder.

## Reproducibility Contract

The deterministic part of this solution starts from a BDD base prediction
folder named by `LTXHF_BASE`. To rebuild the candidate byte-for-byte, use the
same:

1. Track 5 index files, especially `test_index.csv`.
2. LTXHF-T55 BDD base prediction folder.
3. Script parameters in `scripts/build_multiframe_ltxhf_t55_candidate.sh`.
4. WTS route parameters in the same script.

Given identical `LTXHF_BASE` frames and the same index, the multiframe renderer,
hybrid assembly, verification, and zipping steps are deterministic.

If you regenerate the LTXHF-T55 base predictions from a diffusion/video model,
small differences can occur across checkpoints, seeds, schedulers, CUDA kernels,
and hardware. In that case the final candidate will follow the same pipeline but
may not be byte-identical to the tracked submission.

## Method Overview

### 1. BDD base route

The solution expects a BDD-only generated future-frame directory:

```text
/workspace/experiments/wan_ticft/ltxhf_t55_bdd_test/
  report.csv
  video1170/
    0.png
    1.png
    ...
  video135/
    0.png
    1.png
    ...
```

Each BDD test `source_folder` must contain exactly `target_N` numbered PNG
frames. This base route supplies the appearance and dynamic-object predictions.

### 2. Multiframe depth-layered BDD renderer

For each BDD sample, the renderer:

1. Reads the observed history frames from `test_index.csv`.
2. Selects up to eight historical frames with stride two.
3. Estimates depth with `Intel/dpt-hybrid-midas`.
4. Finds actor masks with TorchVision DeepLabV3.
5. Estimates camera motion using ORB matches and PnP.
6. Forward-splats static historical pixels into each future frame with a
   z-buffer.
7. Blends confident static regions into the LTXHF base prediction.
8. Preserves actor and unreliable regions from the base prediction.

Validated parameters:

```text
alpha                  0.75
max_history            8
history_stride         2
base_mode              video
focal_scale            0.9
point_stride           2
splat_radius           1
min_matches            20
min_inliers            12
photo_sigma            32
pose_smoothing_window  7
filled_pose_scale      0.35
confidence_ema         0.65
```

### 3. WTS route

For WTS samples, the wrapper generates or reuses:

- `wts_overhead_or_fixed_camera`: `actor_layer_small_filled`
- `wts_vehicle_or_ip_camera`: `region_norm_medium`

These two routes are written into one WTS output directory and then selected by
view type during hybrid assembly.

### 4. Hybrid assembly

The final candidate uses:

- BDD samples from `depth_multiframe_mf_h8_ltxhf_t55_bdd_test`.
- WTS samples from `wts_overhead_object_layer_vehicle_medium_test`.

The assembler verifies every sample folder and writes a CSV report recording
which source was used for each output.

## Environment

The original experiments used a Linux workspace and a Conda environment named
`wan22`.

Recommended setup:

```bash
conda create -n wan22 python=3.10 -y
conda activate wan22
pip install -r multiframe_ltxhf_t55_solution/requirements.txt
```

The first BDD render will download/cache:

- `Intel/dpt-hybrid-midas`
- TorchVision DeepLabV3 ResNet50 weights

Use a CUDA GPU for the BDD depth/segmentation pass. WTS generation is CPU-only.

## Expected Workspace Layout

The scripts use this layout by default:

```text
/workspace/
  AICITY_Track_5/
    multiframe_ltxhf_t55_solution/
  experiments/
    index/
      test_index.csv
    wan_ticft/
      ltxhf_t55_bdd_test/
      depth_layered_models/
```

You can override every important path with environment variables.

## Quick Start

Clone the repository and enter it:

```bash
cd /workspace
git clone https://github.com/Khangle-2006/AICITY_Track_5.git
cd AICITY_Track_5
```

Install dependencies:

```bash
conda activate wan22
pip install -r multiframe_ltxhf_t55_solution/requirements.txt
```

Build the candidate:

```bash
CUDA_VISIBLE_DEVICES=0 \
INDEX=/workspace/experiments/index \
LTXHF_BASE=/workspace/experiments/wan_ticft/ltxhf_t55_bdd_test \
bash multiframe_ltxhf_t55_solution/scripts/build_multiframe_ltxhf_t55_candidate.sh
```

Default outputs:

```text
/workspace/experiments/wan_ticft/depth_multiframe_mf_h8_ltxhf_t55_bdd_test
/workspace/experiments/wan_ticft/wts_overhead_object_layer_vehicle_medium_test
/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate
/workspace/experiments/wan_ticft/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv
```

Create the submission zip:

```bash
bash multiframe_ltxhf_t55_solution/scripts/create_submission_zip.sh
```

Default zip:

```text
/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate.zip
```

## Path Overrides

Use environment variables when your workspace differs:

```bash
PYTHON=/opt/conda/envs/wan22/bin/python \
INDEX=/data/aicity_track5/index \
TEST_CSV=/data/aicity_track5/index/test_index.csv \
ROOT=/workspace/experiments/wan_ticft \
LTXHF_BASE=/data/predictions/ltxhf_t55_bdd_test \
BDD_RENDERED=/data/predictions/depth_multiframe_mf_h8_ltxhf_t55_bdd_test \
WTS=/data/predictions/wts_overhead_object_layer_vehicle_medium_test \
SUBMISSION=/data/submissions/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate \
REPORT=/data/submissions/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv \
CUDA_VISIBLE_DEVICES=0 \
bash multiframe_ltxhf_t55_solution/scripts/build_multiframe_ltxhf_t55_candidate.sh
```

To zip a non-default folder:

```bash
SUBMISSION=/data/submissions/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate \
ZIP_PATH=/data/submissions/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate.zip \
bash multiframe_ltxhf_t55_solution/scripts/create_submission_zip.sh
```

## Validate The Reference Report

Check the committed reference report:

```bash
python multiframe_ltxhf_t55_solution/scripts/validate_candidate_report.py
```

Expected output includes:

```text
samples=71
frames=5194
statuses={'ok': 71}
ok
```

Validate a newly generated report:

```bash
python multiframe_ltxhf_t55_solution/scripts/validate_candidate_report.py \
  /workspace/experiments/wan_ticft/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv
```

## Optional: Regenerate A BDD Base Folder

`wan_ticft/evaluation/run_ltxv_test_bdd.py` is included as a helper for
generating BDD base videos with an LTX-Video style pipeline. The exact tracked
`ltxhf_t55` base depends on the model checkpoint, prompt recipe, seed, scheduler,
and extraction settings used in the experiment.

Use this helper only if you intentionally want to regenerate the base. For exact
reproduction of the tracked candidate, reuse the original `LTXHF_BASE` frames.

## Submission Folder Requirements

The final submission directory must contain one folder per test output name:

```text
hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate/
  video1170/
    0.png
    ...
  video135/
    0.png
    ...
```

Every sample must contain exactly `target_N` frames from `test_index.csv`.
`baseline.submission` verifies this before zipping.

## Troubleshooting

### `Missing input: .../ltxhf_t55_bdd_test/report.csv`

Set `LTXHF_BASE` to the precomputed BDD base prediction folder. The folder must
contain `report.csv` and per-video PNG subfolders.

### CUDA out of memory

Use a smaller batch for actor segmentation:

```bash
CPU_THREADS=2 CUDA_VISIBLE_DEVICES=0 bash multiframe_ltxhf_t55_solution/scripts/build_multiframe_ltxhf_t55_candidate.sh
```

If needed, edit `--segment_batch 4` in the build script to `2`.

### WTS already exists

The wrapper reuses `WTS` if the directory exists. Delete or change `WTS` if you
want to regenerate the WTS route.

### Score differs from the reference candidate

First compare the generated hybrid report with the committed reference report.
If source choices and frame counts match, the usual cause is a different
`LTXHF_BASE` folder or a regenerated generative base.

## Notes

- This package does not use ground-truth future frames at inference time.
- Generated data and model caches should stay outside Git.
- The multiframe renderer mainly improves static geometry and road/background
  stability; moving-object behavior still comes from the BDD base generator.
