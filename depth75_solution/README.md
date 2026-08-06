# GeoRoute / LTXHF-T55 Reproduction

This directory contains the merged code path for the AI City Challenge 2026
Track 5 submission labeled `ltxhf_t55`. The final system combines the original
Depth75 renderer with the later LTXHF-T55 and multi-frame `mf_h8` improvements.

The submitted system ranked fifth on the official full-test leaderboard:

| Score | PSNR | SSIM | LPIPS | CLIP | FID | FVD |
|---:|---:|---:|---:|---:|---:|---:|
| **73.2843** | 19.7360 | 0.6472 | 0.2942 | 0.9454 | 33.6081 | 29.3483 |

During the challenge, the same candidate first appeared as `69.3815` on the
provisional public subset. The `73.2843` row is the later full-test evaluation
of that submitted route.

## Pipeline

The leaderboard candidate is not a pure LTX-Video output. It is assembled as:

```text
BDD observed clips
  -> LTX/CNN appearance base
  -> traffic-prompt LTX high-frequency donor (alpha 0.55)
  -> LTXHF-T55 BDD base
  -> mf_h8 multi-frame depth/pose/static splatting
  -> refined BDD futures

WTS observed clips
  -> overhead/fixed: actor_layer_small_filled
  -> vehicle/IP: region_norm_medium

refined BDD + routed WTS
  -> 71 folders / 5,194 PNG frames
  -> verification
  -> submission ZIP
```

The `T55` suffix means that high-frequency residuals from the traffic-prompt
LTX donor are transferred with strength `0.55`. The multi-frame renderer then
uses up to eight observed history frames to stabilize confident static regions.
Dynamic actors and unreliable projections remain supplied by the base video.

## Repository Layout

```text
depth75_solution/
  baseline/                         Track 5 I/O, WTS predictors, verification
  reports/
    hybrid_multiframe_...csv        Reference source-selection report
    official_73.2843_sha256.txt     SHA-256 for all 5,194 submitted PNGs
  scripts/
    run_full_pipeline.sh            Recommended end-to-end entrypoint
    setup_ltx_video.sh              Install the official LTX-Video runtime
    relocate_test_index.py          Make the committed index portable
    build_multiframe_...sh          Deterministic mf_h8 + WTS build
    create_submission_zip.sh        Verify and package the final output
    validate_candidate_report.py    Check sample/frame/source counts
    verify_exact_submission.py      Compare output with official SHA-256
    build_depth75_candidate.sh      Legacy single-frame 69.2456 route
  wan_ticft/evaluation/
    run_ltxv_test_bdd.py            BDD LTX-Video extension
    ltx_detail_donor_test.py        Materialize LTXHF-T55 frame folders
    materialize_depth_layered_multiframe.py
    materialize_depth_layered_multiframe_test.py
    materialize_depth_layered_renderer.py
    build_hybrid_submission.py
    ltxv_2b_gate.yaml
  requirements.txt
  test_index_hybrid.csv             View labels and expected frame counts
```

## Reproducibility Levels

There are two supported workflows. The distinction matters for any stochastic
video diffusion system.

### 1. Exact candidate replay

Use the archived `ltxhf_t55_bdd_test` BDD base. All later stages are
deterministic given this folder, the same observed Track 5 frames, model
versions, and index. The final output can be checked byte-for-byte against
`reports/official_73.2843_sha256.txt`.

This is the route to use when auditing the official submission.

### 2. Full model regeneration

Generate the BDD base and traffic-detail donor again with the public
LTX-Video 2B checkpoint, then run T55 and the same deterministic downstream
pipeline. This reproduces the complete method from observed clips, but newly
sampled diffusion frames are not guaranteed to be byte-identical across CUDA,
PyTorch, LTX-Video revisions, schedulers, or hardware.

The official score itself can only be recomputed by the challenge evaluation
server because test future ground truth is not included in this repository.

## Requirements

- Linux
- Python 3.10
- NVIDIA GPU with at least 24 GB VRAM; the original run used one RTX A6000
  with 48 GB
- CUDA-compatible PyTorch
- Approximately 80 GB of free space for data, models, caches, intermediates,
  and the final submission
- AI City Track 5 test data organized as one `<source_folder>/input` directory
  per sample

The geometry and assembly environment can be created with Conda:

```bash
git clone https://github.com/Khangle-2006/AICITY_Track_5.git
cd AICITY_Track_5

conda create -n georoute python=3.10 -y
conda activate georoute
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r depth75_solution/requirements.txt
```

The first geometry run downloads and caches:

- `Intel/dpt-hybrid-midas`
- TorchVision DeepLabV3-ResNet50 weights

## Prepare the Dataset Index

The committed index contains the official 71-sample split and expected future
frame counts. Its original absolute paths are intentionally rewritten for each
machine.

Assume the extracted dataset is under `/data/Track_5`:

```bash
cd AICITY_Track_5
mkdir -p work/index

python depth75_solution/scripts/relocate_test_index.py \
  --template depth75_solution/test_index_hybrid.csv \
  --data-root /data/Track_5 \
  --output work/index/test_index.csv
```

The command searches below `--data-root` for every indexed
`<source_folder>/input` directory and fails on missing or ambiguous samples.

## Exact Replay from the Archived BDD Base

Expected BDD base layout:

```text
/data/artifacts/ltxhf_t55_bdd_test/
  report.csv
  video1170/
    0.png
    1.png
    ...
  ...
```

Run:

```bash
cd AICITY_Track_5
conda activate georoute

MODE=exact \
INDEX_DIR="$PWD/work/index" \
DATA_ROOT=/data/Track_5 \
WORK_ROOT="$PWD/work" \
LTXHF_BASE=/data/artifacts/ltxhf_t55_bdd_test \
CUDA_VISIBLE_DEVICES=0 \
bash depth75_solution/scripts/run_full_pipeline.sh
```

The entrypoint performs the following operations:

1. Runs the `mf_h8` multi-frame renderer on all 44 BDD samples.
2. Generates both WTS specialist routes for 27 WTS samples.
3. Assembles and verifies 71 output folders containing 5,194 PNG frames.
4. Creates the final submission ZIP.

Default outputs:

```text
work/wan_ticft/depth_multiframe_mf_h8_ltxhf_t55_bdd_test/
work/wan_ticft/wts_overhead_object_layer_vehicle_medium_test/
work/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate/
work/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate.zip
```

## Regenerate from the Public LTX Checkpoint

Install the official LTX-Video runtime in a separate virtual environment:

```bash
cd AICITY_Track_5
bash depth75_solution/scripts/setup_ltx_video.sh
```

The setup script clones
[`Lightricks/LTX-Video`](https://github.com/Lightricks/LTX-Video), installs its
inference extras, and downloads
`ltxv-2b-0.9.6-distilled-04-25.safetensors` from the official Hugging Face
repository.

Run the complete regeneration path:

```bash
cd AICITY_Track_5
conda activate georoute

MODE=regenerate \
INDEX_DIR="$PWD/work/index" \
DATA_ROOT=/data/Track_5 \
WORK_ROOT="$PWD/work" \
LTX_REPO="$PWD/depth75_solution/third_party/LTX-Video" \
LTX_PYTHON="$PWD/depth75_solution/third_party/ltxv_env/bin/python" \
PIPELINE_CONFIG="$PWD/depth75_solution/wan_ticft/evaluation/ltxv_2b_gate.yaml" \
SEED=17 \
CUDA_VISIBLE_DEVICES=0 \
bash depth75_solution/scripts/run_full_pipeline.sh
```

This mode generates a default LTX continuation and a traffic-detail donor for
each BDD sample, transfers donor high frequencies with `alpha=0.55`, and then
runs the same `mf_h8`, WTS, assembly, verification, and packaging stages.

All stages support path overrides through environment variables. Run
`bash depth75_solution/scripts/run_full_pipeline.sh --help` for the minimum
interface.

## Validated Parameters

### LTX-Video 2B

| Parameter | Value |
|---|---:|
| Native resolution | 1216 x 704 |
| Output resolution | 1280 x 720 |
| FPS | 30 |
| Inference steps | 8 |
| Guidance scale | 1.0 |
| STG scale | 0 |
| Precision | BF16 |
| Sampling | stochastic, checkpoint sampler |
| Compatible sequence length | `8n + 1` |

### LTXHF-T55

| Parameter | Value |
|---|---:|
| Donor | traffic-prompt LTX |
| Detail alpha | 0.55 |
| Gaussian low-frequency sigma | 2.2 |

### Multi-frame geometry refinement

| Parameter | Value |
|---|---:|
| Maximum history frames | 8 |
| History stride | 2 |
| Blend alpha | 0.75 |
| Point stride | 2 |
| Splat radius | 1 |
| Minimum ORB matches | 20 |
| Minimum PnP-RANSAC inliers | 12 |
| Focal scale | 0.9 |
| Pose smoothing window | 7 |
| Photometric sigma | 32 |
| Filled-pose scale | 0.35 |
| Confidence EMA | 0.65 |

### WTS specialists

| View group | Method | Main settings |
|---|---|---|
| Overhead/fixed | `actor_layer_small_filled` | decay 0.985, max flow scale 12, max area 0.025 |
| Vehicle/IP | `region_norm_medium` | decay 0.96, max flow scale 6 |

## Verify an Existing Submission

Structural verification:

```bash
PYTHONPATH=depth75_solution python -m baseline.submission \
  --index_dir work/index \
  --gen_dir work/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate \
  --submission_dir work/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate \
  --verify_only
```

Byte-level verification against the official submitted PNGs:

```bash
python depth75_solution/scripts/verify_exact_submission.py \
  --submission work/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate \
  --manifest depth75_solution/reports/official_73.2843_sha256.txt
```

Success ends with:

```text
status=exact-match files=5194
```

## Smoke Tests

Static checks that do not require the dataset or GPU:

```bash
python -m compileall -q depth75_solution
PYTHONPATH=depth75_solution python -m unittest discover -s depth75_solution/tests
bash -n depth75_solution/scripts/run_full_pipeline.sh
bash -n depth75_solution/scripts/build_multiframe_ltxhf_t55_candidate.sh
bash -n depth75_solution/scripts/setup_ltx_video.sh
```

For a short end-to-end GPU test, create an index containing one BDD sample and
set `TEST_CSV` to that file. The full candidate validator intentionally expects
71 samples and should only be used for the complete test set.

## Notes

- The official SHA-256 manifest records the exact 73.2843 submission currently
  archived by the team; generated images themselves are excluded from Git due
  to their size.
- Do not change resolution, frame extraction offsets, ordering, filenames, or
  the `8n+1` LTX sequence rule.
- The geometry branch corrects reliable static structure. It does not claim to
  recover ground-truth future actor trajectories.
- The legacy single-frame Depth75 entrypoint remains available for historical
  comparison, but it is not the final 73.2843 route.
