# Experiment Log And Decisions

Last updated: 2026-06-21

This log records the key experiments, metrics, and decisions. It is intentionally decision-focused: what passed, what failed, and what should happen next.

## 2026-06-22 - Supervised Depth Residual Gate

Aligned WTS vehicle16 result:

```text
depth_a75       PSNR 23.9924  SSIM 0.9248  AbsDiff 12.2445  FID 9.1745  FVD 22.8578
supervised_gate PSNR 23.9922  SSIM 0.9248  AbsDiff 12.2446  FID 9.1760  FVD 22.8562
```

Decision:

- Failed gate; do not promote or scale this architecture to train100.
- The 43k-parameter bounded multiplier converged almost exactly to identity:
  mean output change versus depth-a75 was only 0.00023/255.
- This does not reject supervised fine-tuning generally. It shows that merely
  rescaling the frozen depth residual has no remaining learnable headroom.
- Any next supervised attempt must predict a genuinely new correction from
  matched base/history/target data, or fine-tune a pretrained video backbone.

## 2026-06-22 - WTS Vehicle/IP Depth Public Result

```text
hybrid_depth_a75_bdd_wts_vehicle_overhead_object_candidate
Score 69.1801  PSNR 18.5774  SSIM 0.6182  LPIPS 0.3347
CLIP 0.9396  FID 47.4125  FVD 31.2886
```

This is 0.0655 below the BDD-depth-only public best (69.2456). FID improved
by 0.4195, but PSNR, SSIM, LPIPS, CLIP, and FVD moved in the wrong direction.

Root cause: the local depth gate used 16/16 ego `vehicle_view` validation
samples, while public test contains 20/20 IP/fixed-camera samples inside the
same `wts_vehicle_or_ip_camera` label. The first supervised train32 pool was
also 32/32 ego vehicle-view. Do not apply the ego depth route to IP cameras.
Future WTS learning gates must explicitly split ego and IP/normal subdomains.

## Leaderboard State

Current best known public:

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

Public delta versus the previous 69.1436 best: +0.1020 overall, +0.0156
PSNR, unchanged SSIM, +0.0017 LPIPS (worse), +0.0007 CLIP, -0.6709 FID
(better), and +0.0096 FVD (slightly worse). This confirms the BDD depth layer
generalizes, but the gain is incremental rather than a path to 80 by itself.

Main blocker:

- CNN-like outputs increase PSNR/SSIM but hurt FID/FVD and CLIP.
- LTX-like outputs improve distribution but lower pixel metrics and have early ghost/blur.
- WTS remains weak and small hand-engineered improvements only move public score slightly.

## Active Candidate

### temporal latent feature refiner

Heldout8:

```text
cnn_ltxmix65              PSNR 17.8461  SSIM 0.7193  FID 7.4616  FVD 25.8754
temporal_feature_single   PSNR 18.1969  SSIM 0.7237  FID 7.5998  FVD 25.6957
```

Gate24 val-overfit:

```text
cnn_ltxmix65        PSNR 18.2129  SSIM 0.7877  AbsDiff 22.1611  FID 4.9677  FVD 22.2935
temporal_feature    PSNR 18.7337  SSIM 0.7990  AbsDiff 19.9969  FID 5.5718  FVD 23.1755
```

Decision:

- Submit once for public signal if useful.
- Main next work is train-on-train validation, not val-overfit.

Prepared train-on-train command:

```bash
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/run_temporal_latent_feature_train100_gate24.sh
```

Pass criteria:

- BDD24 PSNR +0.15 or better.
- SSIM not lower.
- FVD proxy not worse than `cnn_ltxmix65`.

## Closed Directions

### Alignment/color

```text
baseline temporal_+0      16.9166 PSNR
rgb_affine_crossfit       17.3059 PSNR
delta                     +0.3893
```

Decision: under threshold; not a major bug.

### Multi-history robust flow

```text
region_norm_medium            25.4676 PSNR
region_multi_history_median   25.2219 PSNR
region_multi_history_recency  25.2219 PSNR
```

Decision: failed; closed.

### History camera-motion blend

Full125:

```text
region_norm_medium        26.7591 PSNR  0.9275 SSIM
region_affine_g20_history 26.8733 PSNR  0.9302 SSIM
```

Decision: minor useful idea but not breakthrough.

### Oracle warp/inpaint

Oracle BDD full24:

```text
oracle_splat_repeat_hole_d4 20.7301 PSNR
oracle_splat_target_hole_d4 22.4536 PSNR
```

Decision: oracle confirms headroom; deployable attempts did not work enough.

### Epona

One-frame gate:

```text
delta_psnr +0.3226
delta_ssim +0.0061
wins 3/4
```

Rollout:

```text
delta_psnr -0.4491
delta_ssim -0.0794
worst_delta_psnr -2.1820
```

Decision: rollout/keyframe hybrid closed.

### Cosmos3

Image anchor looked visually promising but did not preserve scene reliably and hurt public:

```text
cosmos public score 68.8484
PSNR 18.9949
SSIM 0.6245
FID 51.2817
FVD 31.3910
CLIP 0.9347
```

5-frame V2V changed scene/cabin.

Decision: closed unless official video conditioning can be made exact.

### Wan2.2 / VACE

Wan2.2 smoke:

```text
cnn_ltxmix65  PSNR 17.1548  FID 9.2509  FVD 29.3045
wan22_ti2v    PSNR 15.5126  FID 9.9896  FVD 35.9703
```

VACE smoke:

```text
cnn_ltxmix65  PSNR 17.1548  FID 9.2509  FVD 29.3045
vace_s80      PSNR 16.7690  FID 10.0079 FVD 32.7988
```

Decision: closed.

### RGB/frame adapters

Residual adapter:

```text
cnn_ltxmix65     PSNR 17.8461  FID 7.4635  FVD 25.8752
metric_adapter   PSNR 18.2798  FID 8.7756  FVD 28.6691
```

Feature/distribution/adversarial adapters also worsened proxy.

Decision: closed output RGB adapters. They overfit pixel and damage feature distribution.

### Temporal denoise

```text
cnn_ltxmix65  PSNR 18.2129  FID 4.9683  FVD 22.2930
td variants   PSNR ~18.00   FID ~5.10    FVD ~22.9+
```

Decision: closed.

### LTX sampler/STG

```text
cnn_ltxmix65          PSNR 17.1548  FID 9.2509  FVD 29.3045
ltx_base              PSNR 16.1623  FID 10.0751 FVD 36.7520
stg_attention_s020    PSNR 16.2020  FID 9.9297  FVD 36.6834
deterministic_decode  PSNR 15.4832  FID 10.9282 FVD 38.9924
```

Decision: closed.

### Rerank/router

Supervised router:

```text
supervised_routed PSNR 18.9206 SSIM 0.8018 FID 7.5922 FVD 27.9924
mix65             PSNR 18.2129 SSIM 0.7877 FID 4.9677 FVD 22.2935
```

Decision: closes as main path; too CNN-like and distribution-bad.

### Low-frequency compositing

Improved pixel but worsened proxy and did not improve public.

Decision: closed.

### WTS object/guard/postprocess

Overhead small object layer full90:

```text
current                FID 3.0361 FVD 16.3848
overhead_small_filled  FID 3.0510 FVD 15.8448
```

Public overhead:

```text
score 68.3523
FID 49.7068
FVD 31.6744
```

Decision:

- Keep overhead object-layer WTS component for candidate builds.
- Do not keep tweaking WTS hand-engineered masks unless a new trainable/model-level route is introduced.

## Commands Worth Keeping

Build current temporal candidate:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/build_temporal_latent_feature_wts_overhead_candidate.sh
```

Run train100 real validation:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/run_temporal_latent_feature_train100_gate24.sh
```

Copy verified candidate:

```cmd
robocopy S:\experiments\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate /E /MT:16 /R:2 /W:2
```

Zip locally:

```cmd
powershell Compress-Archive -Path G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate\* -DestinationPath G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate.zip -Force
```

## Notes For Next Agent

- Do not assume `output.txt` path. Ask for exact path if needed.
- Do not repeat leaked HF token. Use env vars.
- Always verify candidate folders before asking user to submit.
- Visual QA matters. Inspect contact sheets before trusting proxy.

## 2026-06-20 - Temporal Latent Train-On-Train Gate

Goal:

Run the prepared train-on-train temporal latent feature gate before considering
train300 scale-up.

Data issue found:

- The prepared script originally points at
  `/workspace/experiments/wan_ticft/manifests/train_ticft_balanced_300_frames.jsonl`.
- That manifest references `/workspace/experiments/wan_ticft/frames_balanced_300`,
  but that frame directory is no longer present in the workspace.
- Added `wan_ticft/data/materialize_manifest_frames.py` to rematerialize
  condition/target PNGs from the source train videos.
- The balanced-300 manifest contains only `60` BDD rows, not 100. The 121f
  manifest has 100 BDD rows but its BDD target indices overlap the condition
  frames and are not suitable for this future-prediction refiner gate.

Gate result:

```text
cnn_ltxmix65        PSNR 18.2129  SSIM 0.7877  AbsDiff 22.1611  FID 4.9677  FVD 22.2935
temporal_train100   PSNR 18.0991  SSIM 0.7852  AbsDiff 22.3894  FID 5.2649  FVD 22.5984
```

Decision:

- Gate failed: PSNR decreased, SSIM decreased, and FVD proxy worsened.
- Do not scale this run to train300.
- Do not promote the temporal train-on-train refiner.

Visual finding:

Comparison sheets show no catastrophic artifact, but the refiner applies small
residuals that often darken/soften or shift already-good `cnn_ltxmix65` frames.
It helps a few ghosty samples but hurts most BDD24 rows.

Likely causes:

- The training base is region-motion, while validation/test base is
  `cnn_ltxmix65`; the learned residual does not transfer cleanly.
- The prepared train materialization/video-offset path is brittle for sparse
  train targets, so the effective train set is smaller than intended.

Next action:

If temporal refiner is revisited, first generate a BDD train base that is close
to eval/test `cnn_ltxmix65`, then train/evaluate against that matched base.
Do not tune losses or increase train size while the base mismatch remains.

## 2026-06-24 - LTX 2B FrameInit Smoke

Goal:

Test a training-free FrameInit-style latent initialization for LTX 2B: keep
random high-frequency noise, but inject low-frequency structure from the last
observed conditioning latent into future latent frames. The intended upside was
better scene preservation without fully turning the output into repeat-last.

Patch:

- Added opt-in env support in
  `/workspace/third_party/LTX-Video/ltx_video/pipelines/pipeline_ltx_video.py`.
- Disabled by default; only active when `LTX_FRAMEINIT=1`.
- Smoke script:
  `/workspace/run_ltx2b_frameinit_bdd24_gate.sh`.

Smoke result, BDD2:

```text
depth_a75     PSNR 16.4793  SSIM 0.7663  AbsDiff 24.7860  FID  8.3740  FVD 22.7012
ltx2b_base    PSNR 16.3091  SSIM 0.7598  AbsDiff 25.4941  FID  8.8063  FVD 25.5961
frameinit     PSNR 14.2346  SSIM 0.6632  AbsDiff 38.0648  FID 11.9568  FVD 39.8903
repeat_last   PSNR 15.6776  SSIM 0.6813  AbsDiff 28.0577
```

Output:

```text
/workspace/experiments/wan_ticft/ltx2b_frameinit_s035_k9_bdd2_eval.csv
/workspace/experiments/wan_ticft/ltx2b_frameinit_s035_k9_bdd2_proxy.csv
/workspace/experiments/wan_ticft/ltx2b_frameinit_s035_k9_bdd2_compare_sheets
```

Decision:

- Hard fail. Do not run `LIMIT=24` for `s035_k9`.
- Do not submit or scale this FrameInit variant.
- The low-frequency latent injection appears to damage both pixel fidelity and
  temporal/perceptual distribution, exactly the failure mode this project is
  trying to avoid.
- Leave the patch as opt-in only for audit/reproducibility; default behavior is
  unchanged.

## 2026-06-25 - Vista IMG Last-Frame BDD4 Gate

Goal:

Test whether Vista, a driving-specific world model, can serve as a high-realism
BDD candidate using official `IMG` mode from the last condition frame. This is
a feasibility smoke, not a final integration with full history conditioning.

Setup:

- Repo: `/workspace/third_party/Vista`
- Env: `/workspace/experiments/wan_ticft/vista_env`
- Checkpoint: `/workspace/third_party/Vista/ckpts/vista.safetensors`
- Gate script: `/workspace/wan_ticft/evaluation/run_vista_img_bdd4_gate.sh`
- Wrapper: `/workspace/wan_ticft/evaluation/run_vista_img_bdd_gate.py`

Result, BDD4:

```text
depth_a75     PSNR 16.2962  SSIM 0.7610  AbsDiff 26.3280  FID  8.6556  FVD 31.7490
ltx2b_base    PSNR 16.1623  SSIM 0.7556  AbsDiff 26.9257  FID  8.9253  FVD 33.3218
vista_img     PSNR 15.8087  SSIM 0.7409  AbsDiff 27.6220  FID 11.3147  FVD 33.2921
repeat_last   PSNR 15.6917  SSIM 0.7058  AbsDiff 28.1382
```

Outputs:

```text
/workspace/experiments/wan_ticft/vista_img_bdd4_s30_eval.csv
/workspace/experiments/wan_ticft/vista_img_bdd4_s30_proxy.csv
/workspace/experiments/wan_ticft/vista_img_bdd4_s30_compare_sheets
```

Decision:

- Reject raw Vista `IMG` last-frame route. It does not improve FID/FVD and loses
  pixel metrics versus both `depth_a75` and `ltx2b_base`.
- Do not run BDD24 or build a submission from raw `vista_img`.
- Vista may still be useful only with deeper full-history/state conditioning,
  but that is not a quick route. Move now to temporal VSR/detailing on the
  existing `depth_a75` scaffold.

## 2026-06-25 - StableVSR Detail On Depth-A75 BDD4

Goal:

Test temporal diffusion video super-resolution/detailing on top of the current
`depth_a75` scaffold. This targets FID/FVD/LPIPS while preserving the geometry
and pixel structure of the best current BDD branch.

Pipeline:

- Input: `depth_a75` BDD report.
- Downscale future frames to `320x180`.
- StableVSR x4 restoration.
- Materialize report videos at `1280x720`, with frame 0 as the last condition
  anchor for compatibility with existing evaluators.

Result, BDD4, `STEPS=20`:

```text
depth_a75     PSNR 16.2962  SSIM 0.7610  AbsDiff 26.3280  FID 8.6556  FVD 31.7490
stablevsr     PSNR 16.2464  SSIM 0.7610  AbsDiff 26.7402  FID 8.8793  FVD 30.1516
ltx2b_base    PSNR 16.1623  SSIM 0.7556  AbsDiff 26.9257  FID 8.9253  FVD 33.3218
repeat_last   PSNR 15.6917  SSIM 0.7058  AbsDiff 28.1382
```

Visual findings:

- Contact sheets show StableVSR largely preserves `depth_a75` geometry and
  actor identity.
- It does not introduce obvious hallucinated objects in the BDD4 smoke.
- It slightly changes contrast/texture and can look a bit processed, which
  likely explains the mild FID proxy regression.

Decision:

- Borderline continue, not a promotion.
- Positive: FVD proxy improves materially, and PSNR/SSIM are nearly preserved.
- Negative: FID proxy worsens slightly, and the method will not close the
  rank-1 FID gap by itself.
- Run one less-aggressive variant before any BDD24 scale-up:

```bash
LIMIT=4 STEPS=10 LR_WIDTH=480 LR_HEIGHT=270 CUDA_VISIBLE_DEVICES=2 \
  bash /workspace/wan_ticft/evaluation/run_stablevsr_depth_bdd4_gate.sh
```

- Continue to BDD24 only if the variant keeps the FVD gain while reducing the
  FID penalty.

Follow-up less-aggressive variant, `LR_WIDTH=480 LR_HEIGHT=270 STEPS=10`:

```text
depth_a75     PSNR 16.2962  SSIM 0.7610  AbsDiff 26.3280  FID 8.6556  FVD 31.7490
stablevsr     PSNR 16.3379  SSIM 0.7650  AbsDiff 26.5081  FID 8.8440  FVD 32.7494
ltx2b_base    PSNR 16.1623  SSIM 0.7556  AbsDiff 26.9257  FID 8.9253  FVD 33.3218
repeat_last   PSNR 15.6917  SSIM 0.7058  AbsDiff 28.1382
```

Decision update:

- Do not scale StableVSR to BDD24. The aggressive `320x180_s20` variant
  improved FVD but worsened FID; the less-aggressive `480x270_s10` variant
  improved pixel metrics but lost the FVD gain and still worsened FID.
- StableVSR confirms there is some signal in temporal scaffold detailing, but
  off-the-shelf VSR is not enough for a 75+ route.
- Move to a Track5-trained scaffold-conditioned residual/detail model.

## 2026-06-25 - Prepared LTX2B Depth-Scaffold LoRA Smoke

Prepared the next route after rejecting raw Vista and off-the-shelf StableVSR:
train a small LTX2B IC-LoRA to map `depth_a75` scaffold videos to ground-truth
future target videos. This is domain-adaptive scaffold refinement, not full
video-generator pretraining from scratch.

Added:

- `wan_ticft/evaluation/materialize_manifest_target_report.py`
- `wan_ticft/evaluation/score_ltx_scaffold_lora_validation.py`
- `wan_ticft/evaluation/run_ltx2b_depth_scaffold_lora_bdd_smoke.sh`

Updated:

- `wan_ticft/evaluation/prepare_ltx_ic_lora_dataset.py` now accepts prompt,
  step, rank, validation resolution, and checkpoint interval overrides while
  keeping previous defaults.

Smoke command:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_ltx2b_depth_scaffold_lora_bdd_smoke.sh
```

Gate: continue only if checkpoint samples beat the `depth_a75` scaffold on
FID/FVD proxy or visual LPIPS-like quality while keeping geometry, actor
positions, and PSNR/SSIM close.

Runtime fix:

- First run failed in LTX-Video-Trainer preprocessing because `dataset.json`
  used absolute `media_path` / `reference_path` outside `ic_lora_data`, while
  the current trainer calls `relative_to(dataset_file.parent)`.
- `prepare_ltx_ic_lora_dataset.py` now stages target/reference videos under
  `ic_lora_data/media/...` with symlinks and writes relative paths.
- The runner's text-embedding repair now only copies embeddings when the
  expected condition embedding is missing, so it supports the new relative-path
  dataset layout.

BDD4 validation result after 200-step train:

```text
scaffold  PSNR 16.8121  SSIM 0.7814  AbsDiff 24.0446  FID  9.4618  FVD 17.9846
step0     PSNR 11.3499  SSIM 0.2764  AbsDiff 49.6081  FID 19.4067  FVD 45.4499
step100   PSNR 10.7612  SSIM 0.2190  AbsDiff 54.4214  FID 24.4240  FVD 50.3172
step200   PSNR 10.7035  SSIM 0.2108  AbsDiff 54.8376  FID 24.7332  FVD 46.7189
```

Decision:

- Reject this LTX IC-LoRA scaffold route in its current form.
- `step0` already destroys the scaffold, so the problem is not undertraining;
  the official reference-video validation sampler is not acting as a
  copy-preserving scaffold refiner.
- More steps or larger rank are unlikely to fix the core issue and may worsen
  hallucination.
- Keep the scripts for reproducibility, but do not scale to BDD24 or submission.

Next action:

- Stop GPU-heavy generative training until a cheap oracle/headroom audit says
  routing/mixing can still move metrics.
- Added `wan_ticft/evaluation/materialize_oracle_report.py`, which builds a
  per-sample oracle `report.csv` from existing candidate reports using PSNR,
  SSIM, or AbsDiff. Use it to check whether any candidate pool has enough
  headroom before training another router or refiner.

Oracle/headroom audit on BDD24:

```text
depth_a75  PSNR 18.5623  SSIM 0.8344  AbsDiff 19.0541  FID 5.8550  FVD 19.2549
oracle     PSNR 20.6166  SSIM 0.8516  AbsDiff 15.4971  FID 5.0707  FVD 19.4259
```

The oracle chooses only `depth_a75` and `convex_repeat_depth`; the latter is
effectively repeat/static. All-in `convex_repeat_depth` has much worse FVD
(`25.6481`), so the useful signal is narrow routing, not replacing BDD fully.

Simple phase rule `phase1+phase2 -> repeat/convex, otherwise depth_a75`:

```text
depth_a75  PSNR 18.5623  SSIM 0.8344  AbsDiff 19.0541  FID 5.8550  FVD 19.2549
phase12    PSNR 19.9558  SSIM 0.8397  AbsDiff 17.3814  FID 4.9540  FVD 17.3987
```

Decision:

- This is a real pass on BDD24 proxy: pixel, FID, and FVD all improve.
- Build one test candidate with BDD phase1/2 repeat-depth routing and unchanged
  current-best WTS overhead/object-layer outputs.
- Added:
  - `wan_ticft/evaluation/materialize_repeat_last_test.py`
  - `wan_ticft/evaluation/materialize_phase_rule_test.py`
  - `wan_ticft/evaluation/build_bdd_phase12_repeat_depth_wts_overhead_candidate.sh`

## 2026-06-25 - Depth Alpha Temporal Decay Gate

Tested fixed `alpha=0.75` versus a temporal-decay variant that keeps alpha at
0.75 until frame 20 and decays to 0.35 by frame 80:

```text
depth_a75  PSNR 18.5623  SSIM 0.8344  AbsDiff 19.0541  FID 5.8550  FVD 19.2549
hdecay     PSNR 18.5616  SSIM 0.8344  AbsDiff 19.0596  FID 5.8564  FVD 19.2708
phase12    PSNR 19.9558  SSIM 0.8397  AbsDiff 17.3814  FID 4.9540  FVD 17.3987
```

Decision:

- Reject temporal alpha decay as tested. It is effectively neutral and slightly
  worse than `depth_a75`.
- This suggests the current confidence mask already suppresses most risky
  far-horizon splats; changing global alpha over time does not address the main
  gap.
- Keep `phase12` as the active improvement route.

## 2026-06-25 - Prepared BDD Learned Static Router

Next development step after `phase12`: replace the coarse phase-only rule with
a train-on-train static router that chooses `depth_a75` versus `repeat_last`
from motion/caption/phase features.

Added:

- `wan_ticft/evaluation/train_bdd_static_depth_router.py`
- `wan_ticft/evaluation/run_bdd_static_router_train48_gate24.sh`

Design:

- Train on `bdd_supervised_ltx_depth_train48` paired reports.
- Label a train sample as repeat only when repeat beats depth by an AbsDiff
  margin, default `0.25`; otherwise default to depth.
- Features include phase one-hot, history gaps/horizon, condition-frame
  differences, Farneback flow magnitude, sharpness, and caption speed/action
  keywords.
- Infer on independent BDD24 and compare to `depth_a75`, `repeat`, and
  `phase12`.

Gate:

- Continue only if learned static router beats `phase12` or gives a safer
  FID/FVD tradeoff.

## 2026-06-25 - Phase12 Public Result

Submitted `phase12` candidate:

```text
phase12  General Success
Score 68.3785
PSNR 18.2842
SSIM 0.6163
LPIPS 0.3218
CLIP 0.9448
FID 52.3496
FVD 35.1216
```

Compared with the previous best known `depth_a75 + WTS overhead/object-layer`
public result:

```text
best     Score 69.2456  PSNR 18.5942  SSIM 0.6222  LPIPS 0.3312  CLIP 0.9397  FID 47.8320  FVD 31.2770
phase12  Score 68.3785  PSNR 18.2842  SSIM 0.6163  LPIPS 0.3218  CLIP 0.9448  FID 52.3496  FVD 35.1216
```

Decision:

- Reject `phase12` for public despite BDD24 local proxy passing.
- The repeat/static phase route improves local pixel/proxy but public FID/FVD
  collapses badly, so local BDD24 proxy is not representative for this kind of
  static routing.
- Do not build further public candidates that route large BDD subsets to
  repeat/static based only on phase or a small learned router.
- Keep the previous `depth_a75 + WTS overhead/object-layer` candidate as the
  public best baseline.
- Next work should focus on WTS/domain split or distribution-safe candidate
  generation, not repeat-heavy BDD routing.

## 2026-06-25 - WTS Vehicle/IP Source-Split Candidate Prep

Public WTS vehicle-depth attempt (`hybrid_depth_a75_bdd_wts_vehicle_overhead_object_candidate`)
improved FID slightly but reduced overall score versus the best depth75 baseline.
Inspection of `test_index.csv` shows `wts_vehicle_or_ip_camera` is not one
homogeneous domain:

- 11 rows look like WTS main/IP event clips (`20230929...192.168...` /
  `20231006...192.168...`).
- 9 rows are `normal...event...` clips.

Added sample-level routing support:

- `build_hybrid_submission.py`
  - new `--source_folder_override VIEW=SUBSTRING=DIR=LABEL`
  - override is matched against `source_folder` and takes precedence over
    coarse `--view_override`
  - reports now include `source_folder`
- `build_depth75_wts_vehicle_source_split_candidate.sh`
  - `DEPTH_ROUTE=normal`: only apply WTS vehicle depth-a75 to `normal` rows
  - `DEPTH_ROUTE=event`: apply WTS vehicle depth-a75 to non-normal rows, keep
    normal rows on the WTS base
  - `DEPTH_ROUTE=all`: reproduce the full vehicle-depth routing for sanity

Gate:

- Build `normal` and `event` candidates separately.
- Submit at most one split candidate first, preferably the one with the
  smaller blast radius and cleaner route report.
- If both public scores fail to beat `69.2456`, close WTS vehicle-depth and
  return to WTS overhead/object-layer or a trainable distribution-safe route.

## 2026-06-25 - Built WTS Vehicle Normal-Only Candidate

Command:

```bash
CUDA_VISIBLE_DEVICES=2 DEPTH_ROUTE=normal \
ZIP=/workspace/experiments/hybrid_depth75_wts_vehicle_normal_only_candidate.zip \
bash /workspace/wan_ticft/evaluation/build_depth75_wts_vehicle_source_split_candidate.sh
```

Result:

- Rendered WTS vehicle/IP depth-a75 for all 20 rows successfully.
- Built and verified 71/71 output folders.
- Zip created:
  `/workspace/experiments/hybrid_depth75_wts_vehicle_normal_only_candidate.zip`
  (`5.909 GB` on host).
- Route counts:
  - BDD depth-a75: 44
  - WTS overhead/object-layer base: 18
  - WTS vehicle normal-only depth-a75: 9
- WTS vehicle/IP split:
  - `event_ip`: 11 rows kept on base WTS output
  - `normal`: 9 rows routed to WTS vehicle depth-a75

Submit this normal-only candidate before trying `DEPTH_ROUTE=event`, because
it has the smaller blast radius and directly tests whether the previous public
drop came from applying WTS vehicle depth to the wrong source subtype.

Correction / required gate before trusting the zip:

- The test zip itself cannot be scored locally because test targets are hidden.
- Added a validation analog that measures the same source-split idea on WTS
  validation rows with ground truth:
  - `select_wts_vehicle_source_split_manifest.py`
  - `build_wts_vehicle_source_split_report.py`
  - `run_wts_vehicle_source_split_val_gate.sh`
- This gate compares:
  - `current`
  - `normal_only`
  - `event_only`
  - `all_depth`
- Default validation split is 12 `event_ip` + 12 `normal` WTS vehicle/IP rows.

Run before deciding whether to submit/keep `normal_only`:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 LIMIT_PER_SPLIT=12 \
  bash /workspace/wan_ticft/evaluation/run_wts_vehicle_source_split_val_gate.sh
```

Validation result:

```text
Pixel, 24 WTS vehicle/IP rows:
current      PSNR 25.5910  SSIM 0.9455  AbsDiff 9.6097
normal_only  PSNR 25.4500  SSIM 0.9457  AbsDiff 9.7379
event_only   PSNR 25.5398  SSIM 0.9460  AbsDiff 9.6722
all_depth    PSNR 25.3989  SSIM 0.9462  AbsDiff 9.8004

FID/FVD proxy, 19 complete clips:
current      FID 5.0083  FVD 21.6938
normal_only  FID 5.0402  FVD 22.1595
event_only   FID 5.0502  FVD 22.9802
all_depth    FID 5.0956  FVD 23.4349
```

Decision:

- Reject `normal_only`, `event_only`, and `all_depth` WTS vehicle-depth routes.
- Do not submit `hybrid_depth75_wts_vehicle_normal_only_candidate.zip`.
- The previous WTS vehicle/IP base remains better on validation, and this
  agrees with the earlier public result where full WTS vehicle depth reduced
  total score despite a small FID gain.
- Keep the current best public baseline:
  `hybrid_depth_layered_a75_wts_overhead_object_layer_candidate`.

## 2026-06-25 - WTS Background-Locked Actor Composite Prep

Motivation from manual/green-screen idea:

- For fixed WTS views, background should be stable; only actors should move.
- Full object removal / SA2VA / SAM2 may help, but first test a cheap
  version with existing motion masks to see whether background locking helps
  at all.

Added two validation candidates to `evaluate_manifest_motion_baselines.py`:

- `bg_locked_region_long24`
  - generate `region_norm_long24`
  - estimate a clean background from history
  - keep only small foreground-like components from the generated frame
  - composite foreground onto locked background
- `bg_locked_actor_small`
  - same idea, but uses `actor_layer_small_filled` as the actor source

Added gate:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 LIMIT_PER_DOMAIN=24 \
  bash /workspace/wan_ticft/evaluation/run_wts_bglocked_overhead_gate.sh
```

Gate rule:

- Continue to larger WTS overhead validation only if one `bg_locked_*`
  candidate improves FID/FVD versus `actor_layer_small_filled` without a
  material PSNR/AbsDiff collapse.
- If both lose, close the cheap green-screen route before spending time on
  SA2VA/SAM2, because mask quality is probably not the main bottleneck.

Smoke result (`LIMIT_PER_DOMAIN=24`, WTS overhead/fixed):

```text
Pixel:
region_long24           PSNR 31.9527  SSIM 0.9718  AbsDiff 3.4959
actor_small             PSNR 32.0267  SSIM 0.9699  AbsDiff 3.3929
bg_locked_long24        PSNR 32.0629  SSIM 0.9710  AbsDiff 3.3591
bg_locked_actor_small   PSNR 32.0316  SSIM 0.9699  AbsDiff 3.3909

FID/FVD proxy, 22 complete clips:
bg_locked_actor_small   FID 1.4547  FVD 8.5643
actor_small             FID 1.4660  FVD 8.5684
region_long24           FID 1.6925  FVD 9.7755
bg_locked_long24        FID 1.5312  FVD 10.5909
```

Decision:

- `bg_locked_actor_small` has a real but tiny positive signal versus
  `actor_small`.
- `bg_locked_long24` has strong pixel metrics but worse FVD; do not scale that
  branch.
- Continue only by scaling `bg_locked_actor_small` to a larger WTS overhead
  validation gate before any submission build.
- Also updated `materialize_wts_distribution_gate.py` to write
  `selected_manifest.jsonl`, and updated `run_wts_bglocked_overhead_gate.sh`
  so proxy scoring uses only selected rows rather than scanning unrelated
  manifest rows and printing many skips.

## 2026-06-25 - BDD Background-Locked Actor Composite Prep

BDD was not tried first because it is an ego-camera domain: background should
move with ego motion/parallax, so a green-screen/background-lock route is much
riskier than WTS overhead/fixed. Still, it is worth measuring rather than
guessing.

Changes:

- Extended `materialize_wts_distribution_gate.py` to accept `bdd` as a view,
  while preserving existing WTS usage.
- Added `run_bdd_bglocked_gate.sh`.

Run:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 LIMIT_PER_DOMAIN=24 \
  bash /workspace/wan_ticft/evaluation/run_bdd_bglocked_gate.sh
```

The gate compares:

- `region_medium`
- `region_long24`
- `actor_small`
- `bg_locked_long24`
- `bg_locked_actor_small`
- `depth_a75` if the existing report is present

Decision rule:

- Only consider a BDD green-screen route if `bg_locked_actor_small` improves
  FID/FVD versus `actor_small` and remains competitive with `depth_a75`.
- If it loses badly to `depth_a75`, keep this route restricted to fixed-camera
  WTS only.

Initial BDD pixel result (`LIMIT_PER_DOMAIN=24`):

```text
region_medium           PSNR 21.1196  SSIM 0.8508  AbsDiff 15.3415
region_long24           PSNR 21.1035  SSIM 0.8509  AbsDiff 15.3183
actor_small             PSNR 21.2312  SSIM 0.8225  AbsDiff 16.6501
bg_locked_long24        PSNR 20.9578  SSIM 0.8236  AbsDiff 16.6042
bg_locked_actor_small   PSNR 21.2357  SSIM 0.8230  AbsDiff 16.6340
depth_a75               PSNR 18.2677  SSIM 0.8150  AbsDiff 19.0792
```

Notes:

- The green-screen variants do not show a strong BDD signal. They are close to
  `actor_small`, but worse than regional flow on SSIM/AbsDiff.
- The FID/FVD proxy failed because `depth_a75` has a different report/timeline
  shape than the full-video candidates materialized inside this gate, causing
  all clips to be skipped.
- Updated `run_bdd_bglocked_gate.sh`:
  - full-video proxy now compares only the newly materialized candidates
  - a separate timestamp proxy compares all reports including `depth_a75`

Final BDD proxy result after script fix (`LIMIT_PER_DOMAIN=24`):

```text
Timestamp proxy, 23 complete clips:
depth_a75               FID 5.8550  FVD 19.2549
region_medium           FID 7.8984  FVD 21.5093
bg_locked_long24        FID 6.2207  FVD 23.9561
actor_small             FID 6.3358  FVD 25.1205
bg_locked_actor_small   FID 6.3441  FVD 25.3862
region_long24           FID 8.3974  FVD 23.4349
```

Decision:

- Reject BDD green-screen/background-locked route.
- `bg_locked_actor_small` is slightly better than `actor_small` in PSNR, but
  it is much worse than `depth_a75` on FID/FVD and worse than regional flow on
  SSIM/AbsDiff.
- Keep BDD branch as `depth_layered_a75`.

## 2026-06-25 - BDD multi-candidate router gate

Goal:

- Test whether a supervised per-sample router can choose between existing BDD
  candidates better than the fixed `depth_layered_a75` route.
- This is the next high-ceiling direction after rejecting raw generator,
  VSR/detailer, phase12, depth-decay, and green-screen/background-lock routes.

Added:

- `wan_ticft/evaluation/combine_jsonl_manifests.py`
- `wan_ticft/evaluation/combine_video_reports.py`
- `wan_ticft/evaluation/run_bdd_candidate_router_train48_eval24.sh`

Gate:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_bdd_candidate_router_train48_eval24.sh
```

This materializes `region_norm_medium`, `region_norm_long24`, and
`actor_layer_small_filled` for BDD train48/eval24, combines them with existing
`depth_a75`, `depth_a00`, and `repeat_last` reports, trains the small supervised
candidate router on train48, and evaluates only held-out eval24.

Decision rule:

- Continue only if `router` beats `depth_a75` on held-out eval24 FID/FVD proxy
  without a large pixel collapse.
- If it fails, do not submit/router this branch; move to a richer router feature
  set or a different test-time ensemble idea.

Result:

- Raw supervised router improved pixels but hurt distribution:
  - `router`: PSNR 20.9913, SSIM 0.8494, AbsDiff 15.9819
  - `router`: FID 9.1962, FVD 23.3101
  - `depth_a75`: FID 6.9180, FVD 19.2549
- The eval oracle showed that the candidate pool still has signal:
  - `oracle`: PSNR 21.0918, SSIM 0.8700, AbsDiff 15.3407
  - `oracle`: FID 6.9161, FVD 17.6891

Guarded route result:

```text
phase0_depth_phase12_repeat_router_medium_guard
Pixel: PSNR 20.6703, SSIM 0.8396, AbsDiff 16.6134
Proxy: FID 6.5014, FVD 18.8617
depth_a75 proxy: FID 6.9180, FVD 19.2549
```

Interpretation:

- The robust part is not the learned router itself.
- The useful distilled BDD rule is:
  - phase 0 -> `depth_a75`
  - phase 1/2 -> `repeat_last`
  - phase 3/4 -> `region_norm_medium`

Added test builder:

- `wan_ticft/evaluation/build_bdd_guarded_phase_region_wts_candidate.sh`

Command:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/build_bdd_guarded_phase_region_wts_candidate.sh
```

This builds a full 71-sample candidate by applying the distilled BDD phase rule
and keeping the current best WTS branch
`wts_overhead_object_layer_vehicle_medium_test`.

## 2026-07-01 - BDD actor-flow deghost gate

Goal:

- Test a higher-ceiling anti-ghost route than `repeat_anchor`: keep `mf_video`
  background/motion, but replace actor-like regions with real anchor pixels
  warped by dense optical flow.

Added:

- `wan_ticft/evaluation/materialize_actor_flow_deghost.py`
- `wan_ticft/evaluation/run_actorflow_deghost_gate.sh`

Gate base:

- `BASE=/workspace/experiments/wan_ticft/basemode_gate/mf_a75_h4_video`
- Val BDD24, same REAL-FID gate as basemode gate.

Raw actor mask variant:

```text
Pixel:
mf_actorflow  PSNR 18.7492  SSIM 0.8339  AbsDiff 17.7231  Wins 16
mf_video      PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  Wins 7

REAL-FID:
mf_actorflow  63.684
mf_video      68.909
a75_single    70.725
```

Visual decision:

- Reject raw actor mask variant despite strong REAL-FID/pixel signal.
- It replaces too many semantic actor regions, including parked/near vehicles,
  causing obvious pasted-car and black/warped occluder artifacts.

Dynamic-only variant:

- Added `--dynamic_only`, using history residual flow after global affine camera
  compensation to restrict replacement to actor components with non-background
  motion.

```text
Pixel:
mf_actorflow_dyn  PSNR 18.7017  SSIM 0.8350  AbsDiff 18.1537  Wins 8
mf_video          PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  Wins 15

REAL-FID:
mf_actorflow_dyn  67.878
mf_video          68.909
a75_single        70.725
```

Visual decision:

- Dynamic-only is gentler and preserves a small REAL-FID gain, but still shows
  visible paste artifacts on close vehicles and does not recover SSIM.
- Do not submit/build test from these actor-flow variants as-is.
- Useful signal remains: actor replacement can improve REAL-FID/PSNR, but needs
  object-level validation/inpainting or a much safer mask before public probing.

## 2026-07-01 - LTX selector oracle and conservative actor-flow

LTX selector ceiling:

- Added `wan_ticft/evaluation/evaluate_mf_ltx_composite_oracle.py`.
- Inputs: existing `mf_video` validation report and pure LTX validation report.
- This is a GT oracle only; it is not deployable.

Result on BDD23 valid samples:

```text
mf                 PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  LTX  0.00%
ltx                PSNR 18.3821  SSIM 0.8303  AbsDiff 20.0062  LTX 100.00%
frame_oracle       PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  LTX  0.00%
block64_oracle     PSNR 18.6873  SSIM 0.8370  AbsDiff 18.3524  LTX 10.09%
block32_oracle     PSNR 18.6959  SSIM 0.8373  AbsDiff 18.3110  LTX 12.44%
block16_oracle     PSNR 18.7141  SSIM 0.8379  AbsDiff 18.2323  LTX 14.98%
pixel_oracle       PSNR 18.8197  SSIM 0.8414  AbsDiff 17.7484  LTX 19.43%
```

Decision:

- Existing LTX is not a useful full-frame source.
- A practical block/frame selector has too little ceiling; only pixel oracle has
  meaningful gain, and that is not deployable.
- Do not build an LTX selector from the current LTX videos.

MF vlight camera postprocess:

- Added `wan_ticft/evaluation/postprocess_video_report.py`.
- Tested blur `0.3`, noise `2`, JPEG `50` on `mf_video`.

```text
Pixel:
mf_raw     PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  Wins 21
mf_vlight  PSNR 18.6453  SSIM 0.8360  AbsDiff 18.6445  Wins  2

REAL-FID:
mf_raw     68.909
mf_vlight  67.503
```

Decision:

- Same tradeoff as older degradation sweeps: FID improves, but pixel metrics
  degrade broadly. Do not submit as a main 75+ route.

Conservative actor-flow:

- Tightened `materialize_actor_flow_deghost.py` dynamic mask filtering with max
  component area, max bbox height, bottom exclusion, and border exclusion.
- Ran dynamic-only with lower strength and smaller mask:
  `actor_strength=0.45`, `mask_dilate=9`, `mask_blur=3.0`,
  `dynamic_residual_thresh=3.5`, `dynamic_min_area=120`,
  `dynamic_max_area_frac=0.008`, `dynamic_max_bbox_height_frac=0.14`,
  `dynamic_exclude_bottom_frac=0.08`.

```text
Pixel:
mf_actorflow_cons  PSNR 18.6821  SSIM 0.8367  AbsDiff 18.4134  Wins 11
mf_video           PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  Wins 12

REAL-FID:
mf_actorflow_cons  68.857
mf_video           68.909
```

Visual:

- Contact sheets under
  `/workspace/experiments/wan_ticft/actorflow_conservative_gate/sheets`.
- The checked best/worst sheets did not show the severe pasted-car/black-block
  artifacts from the earlier raw actor-flow variant.

Decision:

- This is a safe small Pareto-ish improvement, not a 75+ breakthrough.
- Candidate for a low-risk public probe only after deciding whether small probes
  are still worth queue time; otherwise continue searching for a larger
  generator/conditioning route.

Object-validated actor-flow:

- Added component-level validation to `materialize_actor_flow_deghost.py`.
- Validation happens after warping the dynamic anchor mask into each predicted
  frame:
  - reject components with bad valid-warp coverage,
  - reject components with high boundary-ring RGB mismatch,
  - reject too-large/tall/border/bottom components.
- Added test materializer:
  `wan_ticft/evaluation/materialize_actor_flow_deghost_test.py`.

Val settings:

```text
actor_strength=0.65
mask_dilate=11
mask_blur=3.5
dynamic_residual_thresh=2.8
dynamic_min_area=80
dynamic_max_area_frac=0.05
dynamic_max_bbox_height_frac=0.35
dynamic_exclude_bottom_frac=0.0
component_min_area=80
component_max_area_frac=0.035
component_max_bbox_height_frac=0.30
component_exclude_bottom_frac=0.02
component_boundary=15
component_max_boundary_diff=28
component_min_valid_frac=0.92
```

Val result:

```text
Pixel:
mf_actorflow_objv1  PSNR 18.6813  SSIM 0.8367  AbsDiff 18.4151  Wins 11
mf_video            PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267  Wins 12

REAL-FID:
mf_actorflow_objv1  68.527
mf_video            68.909
```

Visual:

- Sheets under `/workspace/experiments/wan_ticft/actorflow_object_gate/sheets`.
- Checked best/worst PSNR deltas; no severe pasted-car/black-block artifact was
  visible in those sheets.

Built test candidate:

```text
BDD object-flow test:
/workspace/experiments/wan_ticft/depth_multiframe_mf_h8_actorflow_objv1_bdd_test

Full hybrid folder:
/workspace/experiments/hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate

Zip:
/workspace/experiments/hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate.zip
G:\AICITY_Submissions\hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate.zip
```

Verification:

- `build_hybrid_submission`: 71 rows, 5194 frames.
- `baseline.submission --verify_only`: 71 ok.
- G: zip opened successfully; 5194 entries; sampled PNG headers ok.

Decision:

- This is still a small public probe, not a proven 75+ route.
- It is stronger than conservative actor-flow on REAL-FID while keeping pixel
  metrics Pareto-ish, so it is reasonable to submit once and compare against
  `mf_h8=69.3427`.

Follow-up variants while waiting for public:

- `objv2`: more aggressive mask/strength plus paste-region color matching.

```text
Pixel:
mf_actorflow_objv2  PSNR 18.6668  SSIM 0.8364  AbsDiff 18.4475  Wins 1
mf_actorflow_objv1  PSNR 18.6813  SSIM 0.8367  AbsDiff 18.4151
mf_video            PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267

REAL-FID:
mf_actorflow_objv2  68.683
mf_actorflow_objv1  68.527
mf_video            68.909
```

Decision: reject `objv2`; color matching plus looser masks loses pixel and
does not beat `objv1` on REAL-FID.

- `objv3_erase`: `objv1` mask plus nearby base-actor ghost inpainting before
  compositing.

```text
Pixel:
mf_actorflow_objv3  PSNR 18.6790  SSIM 0.8365  AbsDiff 18.4213  Wins 4
mf_actorflow_objv1  PSNR 18.6813  SSIM 0.8367  AbsDiff 18.4151
mf_video            PSNR 18.6771  SSIM 0.8365  AbsDiff 18.4267

REAL-FID:
mf_actorflow_objv3  69.105
mf_actorflow_objv1  68.527
mf_video            68.909

Clip8 proxy:
mf_actorflow_objv1  FID 5.7397  FVD 18.0562
mf_video            FID 5.7579  FVD 18.1181
mf_actorflow_objv3  FID 5.7406  FVD 18.1528
```

Decision: reject `objv3_erase`; inpaint/erase worsens REAL-FID and FVD proxy.
Keep `objv1` as the best actor-flow variant.

## 2026-07-01 - 75+ Route Audit: Retrieval And Warp-Synth

Motivation: public top1 is far ahead mainly on LPIPS/FID/FVD, so small
actor-flow/postprocess tweaks are unlikely to reach 75+. Tested two higher
ceiling routes before spending more GPU.

### Train future retrieval

Added:

- `wan_ticft/evaluation/evaluate_future_retrieval_gate.py`
- `wan_ticft/evaluation/audit_manifest_overlap.py`

Source overlap audit on balanced train/val manifests:

```text
bdd: train_sources=20 val_sources=25 overlap=0 val_overlap_rows=0
wts_overhead_or_fixed_camera: train_sources=26 val_sources=40 overlap=0 val_overlap_rows=0
wts_vehicle_or_ip_camera: train_sources=40 val_sources=34 overlap=0 val_overlap_rows=0
```

BDD retrieval gate, same phase, top1 nearest condition-frame match:

```text
repeat_last        N=25 PSNR=19.8163 SSIM=0.8056 AbsDiff=17.4916
retrieval_top1     N=25 PSNR= 8.7773 SSIM=0.1778 AbsDiff=78.1363
retrieval_top1_cm  N=25 PSNR=11.5352 SSIM=0.2026 AbsDiff=53.5115
```

Decision: close naive retrieval/case-based future route. Balanced train/val
sources do not overlap and nearest-neighbor futures are catastrophically wrong.

### End-to-end warp+synthesis predictor

Added:

- `wan_ticft/evaluation/train_bdd_warp_synth_predictor.py`
- `wan_ticft/evaluation/compute_fid_dirs.py`

This trains a differentiable model that predicts backward sampling flow plus a
small RGB residual and optimizes target-frame reconstruction directly, avoiding
noisy optical-flow supervision. Smoke: BDD train_index 1000 rows, 3 epochs,
320x180, model_width 48.

Training:

```text
epoch=1 samples=920 loss=0.111887 l1=0.051245 ssim=0.273306 edge=0.058983 flow=0.003403 residual=0.002408
epoch=2 samples=920 loss=0.100625 l1=0.046116 ssim=0.245408 edge=0.053866 flow=0.001983 residual=0.001030
epoch=3 samples=920 loss=0.100664 l1=0.045926 ssim=0.246655 edge=0.053766 flow=0.001767 residual=0.000658
```

Val BDD25 sparse:

```text
repeat_last  N=25 PSNR=19.8163 SSIM=0.8056 AbsDiff=17.4916
warp_synth   N=25 PSNR=19.5856 SSIM=0.8147 AbsDiff=17.4989
```

REAL-FID on 216 sparse BDD val frames:

```text
warp_synth FID=82.175348
```

Decision: do not scale this CNN warp-synth branch. It improves SSIM slightly
but worsens distribution badly, consistent with older RGB/feature adapter
failures.

## 2026-07-01 - Learned Flow-Confidence Mask Into MF

Motivation: keep all output pixels real/MF-like, and use a learned motion mask
only to replace ghost-prone regions with lightly warped last-frame pixels. This
avoids the RGB-synthesis failure mode from `warp_synth`.

Added:

- `wan_ticft/evaluation/train_bdd_flow_confidence_predictor.py`
- `wan_ticft/evaluation/materialize_flowconf_mf_composite.py`

Trained BDD flow+confidence model on `train_index` BDD rows:

```text
max_train_rows=1000 epochs=3 batch=6 width=320 height=180 model_width=48
epoch=1 samples=920 loss=0.100006 l1=0.044938 ssim=0.245538 edge=0.054136 flow=0.008098 alpha=0.093152
epoch=2 samples=920 loss=0.097551 l1=0.043922 ssim=0.239442 edge=0.053454 flow=0.005650 alpha=0.067824
epoch=3 samples=920 loss=0.100326 l1=0.045118 ssim=0.247238 edge=0.053719 flow=0.006281 alpha=0.065018
```

Raw flow-confidence output on BDD25:

```text
repeat_last  N=25 PSNR=19.8163 SSIM=0.8056 AbsDiff=17.4916
flow_conf    N=25 PSNR=19.6337 SSIM=0.8146 AbsDiff=17.4417
REAL-FID flow_conf = 71.328
```

Decision: do not use raw flow-conf as a standalone BDD route; FID is worse than
MF. Use its alpha map only.

Composite gate: base=`basemode_gate/mf_a75_h4_video`, source=`flow_conf`,
mask=`alpha`, 23 common BDD val samples. Initial sweep:

```text
mf_base            PSNR 18.6771 SSIM 0.8365 AbsDiff 18.4267
flowconf_t04_s035 PSNR 19.2494 SSIM 0.8506 AbsDiff 17.1150 REAL-FID 67.731
flowconf_t04_s065 PSNR 19.2938 SSIM 0.8474 AbsDiff 16.9460 REAL-FID 71.445
flowconf_t07_s065 PSNR 19.1522 SSIM 0.8458 AbsDiff 17.2392 REAL-FID 70.106
```

Fine sweep:

```text
flowconf_t03_s035 PSNR 19.2963 SSIM 0.8515 AbsDiff 17.0308 REAL-FID 68.030
flowconf_t03_s050 PSNR 19.3897 SSIM 0.8518 AbsDiff 16.8113 REAL-FID 71.181
flowconf_t04_s035 PSNR 19.2494 SSIM 0.8506 AbsDiff 17.1150 REAL-FID 67.731
flowconf_t04_s050 PSNR 19.3316 SSIM 0.8508 AbsDiff 16.9155 REAL-FID 70.665
```

Decision: `flowconf_t04_s035` is the current best learned-mask variant. It is
Pareto-positive versus MF on the BDD REAL-FID gate and pixel metrics. Next:
scale train/infer once, then build a BDD-test route with the same composite
params if the scaled gate keeps REAL-FID <= 68.0.

### Scaled train and test candidate

Scaled flow-confidence model:

```text
max_train_rows=2500 epochs=4 batch=6 width=320 height=180 model_width=48
epoch=1 samples=2325 loss=0.099397 l1=0.044681 ssim=0.244149 edge=0.053474 flow=0.009050 alpha=0.089763
epoch=2 samples=2325 loss=0.096817 l1=0.043317 ssim=0.239208 edge=0.052187 flow=0.006769 alpha=0.074429
epoch=3 samples=2325 loss=0.097175 l1=0.043573 ssim=0.239349 edge=0.051838 flow=0.005863 alpha=0.098013
epoch=4 samples=2325 loss=0.095396 l1=0.042764 ssim=0.234218 edge=0.050690 flow=0.005386 alpha=0.133153
```

Raw BDD25 eval:

```text
repeat_last  N=25 PSNR=19.8163 SSIM=0.8056 AbsDiff=17.4916
flow_conf    N=25 PSNR=19.6736 SSIM=0.8159 AbsDiff=17.3846
```

Scaled composite gate (`threshold=0.04 strength=0.35 softness=0.08 blur=2.0`):

```text
flowconf2500_t04_s035  PSNR=19.1543 SSIM=0.8490 AbsDiff=17.5586
REAL-FID=67.296
```

The scaled model has weaker pixel metrics than the 1000-row model but better
REAL-FID, so it is the current submission probe.

Built full candidate:

```text
BDD replacement:
/workspace/experiments/wan_ticft/depth_multiframe_mf_h8_flowconf2500_t04_s035_bdd_test

Full hybrid:
/workspace/experiments/hybrid_multiframe_mf_h8_flowconf2500_t04_s035_wts_candidate

Zip:
/workspace/experiments/hybrid_multiframe_mf_h8_flowconf2500_t04_s035_wts_candidate.zip
```

Verification:

- `build_hybrid_submission`: 71 rows, 5194 frames.
- `baseline.submission --verify_only`: 71 ok.
- Zip opens; 5194 PNG entries; sampled PNG headers ok.

## 2026-07-02 - WTS Vehicle Phase-Routed Flow-Confidence

Motivation: WTS top gap is mostly perceptual/FVD. A WTS vehicle-specific
flow-confidence probe improved target-frame pixels, but applying it to all
vehicle phases improved FID while slightly hurting FVD. Phase proxy showed the
useful signal is concentrated in vehicle phases 1 and 2; phases 0 and 3 should
stay on the current WTS vehicle base.

Added/updated:

- `train_bdd_flow_confidence_predictor.py`: `--view_type` support for WTS.
- `build_wts_view_routed_report.py`: infer missing `view_type`, union fields.
- `materialize_target_aligned_report.py`: compact target-aligned video reports.
- `build_phase_routed_report.py`: route selected phases from an alternate report.
- `materialize_flowconf_submission_composite.py`: phase include + copy fallback.

Gate, target-aligned WTS main validation, current overhead + vehicle:

```text
current       PSNR=29.9741 SSIM=0.9667 AbsDiff=5.9575 FIDp=4.8551 FVDp=14.8648
p12_t02s035   PSNR=30.0937 SSIM=0.9694 AbsDiff=5.8050 FIDp=4.6065 FVDp=14.6606
```

Phase proxy for vehicle showed:

```text
phase0: current better FVD
phase1: t02s035 better FID/FVD
phase2: t02s035 better FID/FVD
phase3: current better FVD
```

Built full test candidate:

```text
/workspace/experiments/hybrid_multiframe_mf_h8_flowconf2500_wts_vehicle_p12flowconf_t02s035_candidate
/workspace/experiments/hybrid_multiframe_mf_h8_flowconf2500_wts_vehicle_p12flowconf_t02s035_candidate.zip
```

Verification:

- `build_hybrid_submission`: 71 rows, 5194 frames.
- `baseline.submission --verify_only`: 71 ok.
- `zipinfo`: 5194 entries / 5194 PNG.
- `unzip -t`: no compressed-data errors.

## 2026-07-02 - BDD Semantic-Validated Actorflow

Motivation: raw actorflow gave a strong REAL-FID signal but had visible pasted
actor / occlusion artifacts. The first conservative object version submitted
slightly positive on public (`69.3579`) but hurt FVD. This sweep looked for a
middle ground: keep real moving actors, but validate components semantically and
geometrically before compositing.

Validation gate, BDD selected set:

```text
Pixel:
mf_video              PSNR=18.6771 SSIM=0.8365 AbsDiff=18.4267 Wins=0
relax_dyn             PSNR=18.6841 SSIM=0.8367 AbsDiff=18.3987 Wins=3
relax_dyn_strong      PSNR=18.6787 SSIM=0.8365 AbsDiff=18.4041 Wins=3
semantic_valid        PSNR=18.6981 SSIM=0.8369 AbsDiff=18.3515 Wins=17

REAL-FID:
a75_single        70.725
mf_video          68.909
relax_dyn         68.822
relax_dyn_strong  68.738
semantic_valid    68.051

Clip8 proxy:
semantic_valid  FIDp=7.1152 FVDp=20.5244
objv1           FIDp=7.3819 FVDp=21.5881
mf_video        FIDp=7.4051 FVDp=21.6283
flowconf2500    FIDp=7.6010 FVDp=21.7850
```

Visual QA sheet:

```text
/workspace/experiments/wan_ticft/actorflow_relaxed_sweep_gate/semantic_valid_sheet.png
```

Built full test candidate:

```text
/workspace/experiments/hybrid_multiframe_mf_h8_actorflow_semantic_valid_wts_candidate
/workspace/experiments/hybrid_multiframe_mf_h8_actorflow_semantic_valid_wts_candidate.zip
```

Verification:

- `build_hybrid_submission`: 71 rows, 5194 frames.
- `baseline.submission --verify_only`: 71 ok.
- Zip opens; 5194 entries / 5194 PNG.
- `unzip -t`: no compressed-data errors.

Assessment: stronger than objv1 on the trusted BDD gate and FVD proxy, so it is
worth a public submit. It is still a candidate probe rather than a credible
standalone route to 75+; the remaining gap is primarily model/perceptual quality.

## 2026-07-02 - Background Smear / Afterimage Audit

User-visible issue: many BDD outputs have semi-transparent background trails and
windshield-like smear. This is a global base-frame artifact, not just the actor
double-image problem. Actor/object compositing cannot remove it if the selected
background is already smeared.

Tested a motion-aligned last-condition-frame texture rescue:

```text
/workspace/wan_ticft/evaluation/materialize_anchor_background_rescue.py
/workspace/experiments/wan_ticft/anchor_background_rescue_gate
```

Variants transfer luma high/mid/low bands from the aligned last real frame into
the `mf_video` base only at confident optical-flow locations.

Gate result, 23 common BDD validation samples:

```text
Pixel:
mf_video    PSNR=18.6771 SSIM=0.8365 AbsDiff=18.4267 Wins=19
yfull_a08   PSNR=18.6713 SSIM=0.8365 AbsDiff=18.4398 Wins=4
yhf_a25     PSNR=18.6440 SSIM=0.8360 AbsDiff=18.6175 Wins=0
ymid_a20    PSNR=18.6441 SSIM=0.8359 AbsDiff=18.6302 Wins=0
ysmear_a12  PSNR=18.6673 SSIM=0.8364 AbsDiff=18.4691 Wins=0

REAL-FID:
mf_video    68.909
yhf_a25     69.017
yfull_a08   69.038
ysmear_a12  69.080
ymid_a20    69.100
```

Decision: reject this simple anchor-rescue route. It confirms the smear is a
base/model quality problem; naive last-frame texture transfer makes REAL-FID
worse. The next credible route must change the BDD generator or train a
target-supervised residual/detail model, not just add post-hoc blur/sharpen or
background warping.

## 2026-07-02 - Distribution Router On MF/Actorflow Pool

Tested no-reference per-sample routing using Inception/R3D feature distance to
the validation GT distribution:

```text
/workspace/wan_ticft/evaluation/rerank_distribution_router.py
/workspace/wan_ticft/evaluation/materialize_video_report_dir.py
/workspace/wan_ticft/evaluation/normalize_frame_folder.py
```

Pool 1: `mf_video`, `semantic_valid`, `flowconf2500`, `objv1`, `depth_a75`.

```text
Choices: semantic_valid=10, objv1=5, depth_a75=4, mf_video=3, flowconf2500=1
Pixel: PSNR=18.7230 SSIM=0.8382 AbsDiff=18.4717
REAL-FID routed=71.352 after frame-size normalization
```

Pool 2, same-base only: `mf_video`, `semantic_valid`, `objv1`.

```text
Choices: semantic_valid=12, objv1=6, mf_video=5
Pixel: PSNR=18.6916 SSIM=0.8367 AbsDiff=18.3770
REAL-FID semantic_valid=68.051
REAL-FID dist_same=70.402
```

Decision: reject distribution-router. The no-reference feature score chooses
plausible-looking mixes but hurts the trusted REAL-FID gate. Current best BDD
gate remains `semantic_valid`; routing among similar MF/actorflow variants does
not create the needed 75+ jump.

## 2026-07-02 - Flow-Confidence 5k Retrain

Motivation: the learned flow-confidence composite was the strongest trusted
BDD REAL-FID gate so far (`flowconf2500_t04_s035 REAL-FID=67.296`). Tested
whether scaling the alpha predictor from 2500 rows / 4 epochs to 5000 rows /
5 epochs improves the composite.

Setup:

```text
train_bdd_flow_confidence_predictor
max_train_rows=5000 epochs=5 batch=6 width=320 height=180 model_width=48
```

Training:

```text
epoch=1 samples=4645 loss=0.099477 l1=0.044688 ssim=0.245141 edge=0.053137 flow=0.007235 alpha=0.074872
epoch=2 samples=4646 loss=0.096515 l1=0.043143 ssim=0.238183 edge=0.051524 flow=0.006259 alpha=0.104205
epoch=3 samples=4645 loss=0.096168 l1=0.043199 ssim=0.235501 edge=0.050480 flow=0.005424 alpha=0.153352
epoch=4 samples=4645 loss=0.094475 l1=0.042294 ssim=0.231408 edge=0.049978 flow=0.005079 alpha=0.170239
epoch=5 samples=4645 loss=0.094101 l1=0.042204 ssim=0.229888 edge=0.049705 flow=0.005085 alpha=0.179440
```

Composite gate on 23 common sparse BDD val samples:

```text
mf_base       PSNR=18.6771 SSIM=0.8365 AbsDiff=18.4267 REAL-FID=68.909
t03_s035      PSNR=19.2476 SSIM=0.8517 AbsDiff=17.3128 REAL-FID=68.087
t04_s035      PSNR=19.2348 SSIM=0.8513 AbsDiff=17.3394 REAL-FID=68.000
t03_s050      PSNR=19.2962 SSIM=0.8510 AbsDiff=17.2739 REAL-FID=72.652
t04_s050      PSNR=19.2810 SSIM=0.8506 AbsDiff=17.3070 REAL-FID=72.583
```

Decision:

- Reject 5k retrain as a new submission route. It improves sparse pixel metrics
  but does not beat the 2500-row model on trusted REAL-FID.
- Do not build a 5k test candidate unless a later public result contradicts the
  REAL-FID ranking.

## 2026-07-02 - LTX Chunk Visual/Prompt Audit

User observation: LTX/chunk outputs do not have the MF-style translucent
afterimage, but they look synthetic/rendered.

Visual check:

- `ltx2b_chunked_traffic_bdd24_gate` and `ltx2b_chunked_traffic_bdd24_c8`
  both preserve cleaner motion than MF/depth but show waxy road/building
  texture, over-clean surfaces, and a generated-image color/contrast profile.
- `ltx2b_chunked_realism_repair_bdd24_gate` does not fix the synthetic look.
- `ltx13b_base_bdd4_gate` and detailer blends still show the same polished
  rendered texture, so switching 2B -> 13B is not sufficient by itself.

Prompt diagnosis:

- Current chunk default `traffic_detail` heavily emphasizes "crisp", "sharp",
  and "detailed asphalt texture". This likely pushes LTX toward synthetic
  beautified texture instead of matching noisy dashcam footage.
- Added prompt mode `dashcam_native` to
  `run_ltxv_val_gate.py` / `run_ltxv_val_chunked_gate.py`. It explicitly asks
  for the same real dashcam recording, lens/exposure/color response, windshield
  reflections, sensor noise, compression texture, natural motion blur, and no
  artificial sharpening/cleanup.
- Added `--sample_ids` support to `run_ltxv_val_chunked_gate.py`.
- Added smoke script:
  `/workspace/wan_ticft/evaluation/run_ltx2b_chunked_prompt_smoke4.sh`.

Next:

Run the smoke when GPU utilization is low:

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_ltx2b_chunked_prompt_smoke4.sh
```

Compare sheets under:

```text
/workspace/experiments/wan_ticft/ltx2b_chunked_prompt_smoke4_<mode>_c16/compare_sheets
```

Promote only if `dashcam_native` or `copy_bias` visibly reduces synthetic
texture without reintroducing afterimage, then run REAL-FID before any
submission build.

## 2026-07-02 - FramePack F1 BDD Smoke

Tested FramePack F1 image-to-video as a model-level route to remove base smear:

```text
/workspace/wan_ticft/evaluation/run_framepack_f1_bdd_smoke.py
/workspace/experiments/wan_ticft/framepack_f1_bdd1_smoke
/workspace/experiments/wan_ticft/framepack_f1_bdd1_shortprompt_s8
```

Environment note: `wan22` has `diffusers=0.38.0` and FramePack pipeline classes;
the old `cosmos25_diffusers_env` path is missing. First run populated
`/workspace/experiments/wan_ticft/framepack_hf_cache`.

BDD1 result on `val_bdd_pc_5k_video7040_phase0`:

```text
mf_video         PSNR=16.3671 SSIM=0.8208 AbsDiff=23.8277
repeat_last      PSNR=14.8636 SSIM=0.7026 AbsDiff=29.5481
framepack_long   PSNR=14.5049 SSIM=0.6878 AbsDiff=30.8421
framepack_short  PSNR=14.6166 SSIM=0.7028 AbsDiff=30.1843
```

Visual sheets:

```text
/workspace/experiments/wan_ticft/framepack_f1_bdd1_smoke_sheets
/workspace/experiments/wan_ticft/framepack_f1_bdd1_shortprompt_s8_sheets
```

Decision: do not scale FramePack F1. Short prompt is slightly better than the
truncated long prompt, but both alter geometry/motion too much and still show
large road smear. This is not a credible 75+ route without additional control or
fine-tuning.

## 2026-07-02 - BDD Refiner Router V2 Audit

Audited the existing direct BDD refiner checkpoint:

```text
/workspace/experiments/refiner_bdd_direct_h1_120_v2/refiner_best.pt
/workspace/wan_ticft/evaluation/materialize_bdd_refiner_router_v2_val.py
/workspace/experiments/wan_ticft/bdd_refiner_router_v2_gate24
```

Pixel gate on BDD24 is very strong:

```text
semantic_valid     PSNR=18.6981 SSIM=0.8369 AbsDiff=18.3515
refiner_router_v2  PSNR=21.4935 SSIM=0.8580 AbsDiff=15.0324
```

Trusted REAL-FID gate:

```text
semantic_valid     REAL-FID=68.051
refiner_router_v2  REAL-FID=120.161
```

Decision: reject. This is an extreme example of local pixel metrics lying. The
refiner produces CNN/warp-like images far outside the real BDD distribution.
Do not build or submit `bdd_refiner_router_v2_sharp_test`.
