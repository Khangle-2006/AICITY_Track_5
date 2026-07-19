# AI City Track 5 Handoff

Last updated: 2026-06-21

This file is the handoff state for the next Codex/chat session. The goal is to let the next agent start working without replaying the whole conversation.

## Goal

We are working on AI City Track 5 future video prediction. The target is an aggressive leaderboard jump, ideally toward 80, but the current best public result is around 69. The user does not want endless tiny tweaks. Keep testing bounded, close weak directions quickly, and prioritize generalizable methods that can work on private data.

The current challenge:

- BDD outputs have become reasonably strong in PSNR/SSIM.
- WTS and distribution metrics remain weak.
- FID/FVD are hard to reduce without losing PSNR/SSIM.
- Private set is unknown, so avoid brittle routing that depends on public/test composition.

## Current Best Known Public Submissions

| Name / description | Public score | PSNR | SSIM | LPIPS | CLIP | FID | FVD | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hybrid_depth_layered_a75_wts_overhead_object_layer_candidate` | 69.2456 | 18.5942 | 0.6222 | 0.3312 | 0.9397 | 47.8320 | 31.2770 | Best known public score. Adds validated depth-layered a75 to BDD; WTS remains unchanged. |
| `hybrid_depth_a75_bdd_wts_vehicle_overhead_object_candidate` | 69.1801 | 18.5774 | 0.6182 | 0.3347 | 0.9396 | 47.4125 | 31.2886 | WTS depth hurt public: local gate was ego vehicle-view, but test WTS vehicle/IP is entirely IP/fixed-camera. |
| `hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate` | 69.1436 | 18.5786 | 0.6222 | 0.3295 | 0.9390 | 48.5029 | 31.2674 | Previous best baseline. |
| `ltx_45_guard015` | 69.0025 | 18.8321 | 0.6245 | 0.3244 | 0.9372 | 50.4442 | 31.7726 | Higher pixel, worse FID/FVD. |
| `cosmos` / Cosmos background hybrid | 68.8484 | 18.9949 | 0.6245 | 0.3276 | 0.9347 | 51.2817 | 31.3910 | Pixel high but CLIP/FID worse. |
| `overhead` | 68.3523 | 18.1614 | 0.6113 | 0.3395 | 0.9357 | 49.7068 | 31.6744 | Small WTS improvement only. |
| `cnn` | 67.6794 | 19.2046 | 0.6229 | 0.3159 | 0.9289 | 58.4865 | 35.2033 | Pixel very high but FID/FVD/CLIP too bad. |

## Current Main Pipeline

Best deployable baseline:

- BDD: `cnn_ltxmix65`
- WTS overhead/fixed: overhead object-layer small-filled candidate
- WTS vehicle/IP: region medium / guard route depending candidate

Important paths:

- Repo: `/workspace/wan_ticft` (`S:\wan_ticft`)
- Experiments: `/workspace/experiments` (`S:\experiments`)
- Index: `/workspace/experiments/index`
- BDD validation manifest: `/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl`
- BDD base report: `/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv`
- Best public candidate folder:
  `/workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate`

Current verified experimental candidate:

- Folder:
  `/workspace/experiments/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate`
- Verified:
  `71 rows / 5194 frames`
- BDD source:
  `/workspace/experiments/wan_ticft/temporal_latent_feature_bdd_test`
- WTS source:
  `/workspace/experiments/wan_ticft/wts_overhead_object_layer_vehicle_medium_test`

To copy that candidate to local:

```cmd
robocopy S:\experiments\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate /E /MT:16 /R:2 /W:2
```

Then zip from local:

```cmd
powershell Compress-Archive -Path G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate\* -DestinationPath G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate.zip -Force
```

The zip must contain sample folders directly, not an extra parent directory.

## Most Promising Direction Now

### Temporal latent feature refiner

This is the only learned refiner so far that produced strong PSNR/SSIM gains without catastrophic FVD drift.

Heldout8 result:

- `temporal_feature_single`: PSNR 18.1969 vs `cnn_ltxmix65` 17.8461
- SSIM 0.7237 vs 0.7193
- AbsDiff 22.3442 vs 24.0874
- FVD proxy 25.6957 vs 25.8754, slightly better
- FID proxy 7.5998 vs 7.4616, slightly worse

Gate24 trained/scored on BDD24:

- `temporal_feature`: PSNR 18.7337 vs `cnn_ltxmix65` 18.2129
- SSIM 0.7990 vs 0.7877
- AbsDiff 19.9969 vs 22.1611
- Wins 19/24
- FID proxy 5.5718 vs 4.9677
- FVD proxy 23.1755 vs 22.2935

Interpretation:

- Very strong pixel gain.
- Proxy distribution worsens, but far less than raw `cnn0`.
- Worth submitting once for public signal.
- Not enough for 80 by itself.

Important scripts:

- Trainer:
  `/workspace/wan_ticft/evaluation/train_temporal_latent_refiner.py`
- Single-GPU feature smoke:
  `/workspace/wan_ticft/evaluation/run_temporal_latent_refiner_feature_single_gpu.sh`
- BDD24 val-overfit gate:
  `/workspace/wan_ticft/evaluation/run_temporal_latent_feature_gate24.sh`
- Test candidate builder:
  `/workspace/wan_ticft/evaluation/build_temporal_latent_feature_wts_overhead_candidate.sh`

### Latest train-on-train gate update

The prepared train100 gate has now been checked and failed.

Data repair was required:

- `train_ticft_balanced_300_frames.jsonl` referenced the missing
  `/workspace/experiments/wan_ticft/frames_balanced_300` directory.
- Added `/workspace/wan_ticft/data/materialize_manifest_frames.py` to
  rematerialize condition/target PNGs from source train videos.
- The balanced-300 manifest has only 60 BDD rows, so the fixed run is
  effectively a smaller train gate rather than true train100.
- The 121f manifest has 100 BDD rows but its BDD target indices overlap the
  condition frames, so it is not suitable for this future-refiner gate.

Gate result:

```text
cnn_ltxmix65        PSNR 18.2129  SSIM 0.7877  AbsDiff 22.1611  FID 4.9677  FVD 22.2935
temporal_train100   PSNR 18.0991  SSIM 0.7852  AbsDiff 22.3894  FID 5.2649  FVD 22.5984
```

Decision:

- Failed all continue criteria.
- Do not scale this run to train300.
- Do not tune losses against the same region-motion train base.
- If temporal refiner is revisited, first create BDD train predictions that
  match the eval/test `cnn_ltxmix65` base, then rerun the gate.

Contact/comparison sheets:

- `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24_sheets`
- `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24_compare_sheets`

Note: a later accidental rerun with the original missing-frame manifest cleaned
the model/report output directory, so trust the CSV/proxy/sheet outputs above
for the completed gate result.

### What should be done next

The real next direction is **not** more val-overfit and not scaling the failed
region-motion train gate. It is train-base matching:

1. Generate a BDD train base that is close to the eval/test `cnn_ltxmix65`
   base, ideally by materializing CNN selector + LTX mix65 on train rows.
2. Train temporal latent feature refiner on that matched BDD train base.
3. Validate on BDD24 against `cnn_ltxmix65`.
4. Continue only if PSNR improves by >= +0.15, SSIM does not drop, and FVD
   proxy does not worsen.
5. Only after a matched-base BDD pass, train a WTS-specific refiner.

Historical scripts:

- Region-motion base materializer:
  `/workspace/wan_ticft/evaluation/materialize_region_motion_report.py`
- Failed region-motion train gate:
  `/workspace/wan_ticft/evaluation/run_temporal_latent_feature_train100_gate24.sh`

Do not rerun the failed gate as the next step. It uses the wrong train base for
the current eval/test branch and the default manifest still points at missing
frame PNGs unless repaired.

## Directions Already Tried And Closed

### Alignment/color audit

Result:

- Best variant `rgb_affine_crossfit`: 17.3059 PSNR vs baseline 16.9166, +0.3893.
- Under +0.5 dB threshold.

Decision:

- No major alignment/color bug. Closed.

### Oracle warp/inpaint

Oracle showed headroom:

- BDD full24 target-hole d4 reached 22.4536 PSNR vs repeat-hole d4 20.7301.
- Cross-domain oracle also had headroom, especially BDD and WTS vehicle.

But deployable flow/inpaint attempts did not generalize or introduced artifacts.

Decision:

- Oracle useful as diagnostic, but no full layered warp/inpaint pipeline.

### Multi-history flow

Smoke and recency-weighted correction failed:

- Macro PSNR lower than `region_norm_medium`.
- WTS vehicle/IP lost badly.

Decision:

- Closed multi-history flow.

### History-derived global camera blend

`region_affine_g20_history` passed small/full validation:

- Full125: PSNR 26.8733 vs 26.7591; SSIM 0.9302 vs 0.9275.
- But overhead domain lost slightly.

Decision:

- Useful minor component, not breakthrough.

### LTX history length/prompt tuning

History17/25/adaptive did not beat history41 robustly.

Prompt tuning:

- Minimal prompt: no improvement.
- Manifest prompt: tiny mixed gain.
- Behavior prompt: small SSIM gain only.

Decision:

- Closed LTX history/prompt tuning.

### Low-frequency correction

BDD gate:

- Pixel improved over LTX16: +0.4783 PSNR, +0.0088 SSIM, AbsDiff improved.
- But proxy worsened and public did not improve.

Decision:

- Do not promote; pixel-only direction.

### BDD transition phase routing

Phase0 route32 improved proxy in local aggregate but public did not improve:

- `transition32` public score stayed around 68.3249.

Decision:

- Closed transition routing.

### LTX masked transition / STG sampler

Masked transition:

- FID/FVD proxy worsened versus transition16.

STG sampler smoke:

- `stg_attention_s020`/`stg_residual_s020` only tiny PSNR gain over LTX base.
- FVD proxy still much worse than `cnn_ltxmix65`.
- Deterministic decode worse.

Decision:

- Closed LTX sampler/transition masks.

### WTS postprocess/texture

Unsharp/CLAHE:

- All worsened FID/FVD proxy.

Decision:

- Closed texture postprocess.

### WTS object-layer / actor guard

Overhead object-layer route:

- Full90 proxy:
  - `overhead_small_filled`: FVD 15.8448 vs current 16.3848, FID 3.0510 vs 3.0361.
- Public `overhead`: small score increase to 68.3523.

Vehicle object-layer:

- Proxy FID sometimes improved but FVD/pixel worsened; visual artifacts remained.

Decision:

- Keep overhead-object candidate as current WTS component.
- Do not continue hand-engineered WTS anti-smear compositing unless new model-level idea appears.

### Epona

One-frame smoke:

- Epona beat repeat-last on one smoke, and 4-sample one-frame gate passed:
  - +0.3226 PSNR, +0.0061 SSIM, 3/4 wins.

Rollout failed:

- Aggregate PSNR -0.4491, SSIM -0.0794.
- Step4 PSNR -1.1183.
- Worst loss -2.1820.

Keyframe hybrid failed badly:

- Hybrid PSNR -3.9 vs region.
- Masked correction still -2.28.

Decision:

- Closed Epona rollout/keyframe hybrid.

### Cosmos2.5 / Cosmos3

Cosmos2.5:

- Gated repo 403. No access.

Cosmos3:

- Direct vLLM-Omni setup eventually ran.
- Image-anchor sometimes looked visually better but had scene/camera issues and public score worse.
- 5-frame V2V changed the scene/cabin; hard fail.
- Background hybrid improved local pixel but worsened FID/CLIP public:
  - public `cosmos`: 68.8484, FID 51.2817, CLIP 0.9347.

Decision:

- Cosmos is visually interesting but not deployable under current API/prompt/conditioning.
- Only revisit if official 5-frame V2V conditioning can be made to preserve exact scene, not as current image-anchor hack.

### Wan2.2-TI2V / VACE

Wan2.2-TI2V smoke:

- PSNR 15.5126 vs cnn_ltxmix65 17.1548.
- FID/FVD proxy worse.

VACE refiner:

- `vace_s80` PSNR 16.7690 vs cnn_ltxmix65 17.1548.
- FID/FVD worse.

Decision:

- Closed model-swap refiner route.

### Output RGB adapters

Metric residual adapter:

- Pixel improved on heldout8:
  - PSNR 18.2798 vs 17.8461.
  - SSIM 0.7225 vs 0.7193.
- But FID/FVD worsened badly:
  - FID 8.7756 vs 7.4635.
  - FVD 28.6691 vs 25.8752.

Feature/distribution/adversarial adapters:

- Failed to improve proxy.

Decision:

- Closed frame-wise/output RGB adapters. They optimize pixel but damage distribution.

### Distribution rerank / supervised router

Simple no-reference distribution reranker:

- Pixel much worse; proxy not better than fixed mix65/mix75.

Supervised router:

- Pixel excellent:
  - PSNR 18.9206, SSIM 0.8018.
- Proxy bad:
  - FID 7.5922, FVD 27.9924.
- Essentially picked CNN-like candidates.

Decision:

- Closed rerank/router as main direction.

## Current/Recent Commands

Build temporal latent feature candidate:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/build_temporal_latent_feature_wts_overhead_candidate.sh
```

Train100 real generalization gate:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/run_temporal_latent_feature_train100_gate24.sh
```

Single-GPU feature refiner smoke:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/run_temporal_latent_refiner_feature_single_gpu.sh
```

Gate24 val-overfit:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/run_temporal_latent_feature_gate24.sh
```

## What The Next Agent Should Inspect

Before making new changes, inspect:

1. Markdown files:
   - `/workspace/wan_ticft/PROJECT_HANDOFF.md`
   - `/workspace/wan_ticft/EXPERIMENT_LOG.md`
   - `/workspace/wan_ticft/NEXT_CHAT_PROMPT.md`
2. Current experiment folders:
   - `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24*`
   - `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24*`
   - `/workspace/experiments/wan_ticft/temporal_latent_feature_bdd_test`
   - `/workspace/experiments/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate`
3. Visual sheets:
   - `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24_sheets`
   - `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24_sheets` if generated.
4. Existing best candidates and reports:
   - `/workspace/experiments/wan_ticft/hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate_report.csv`
   - `/workspace/experiments/wan_ticft/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate_report.csv`

Also inspect any `output.txt` the user mentions by asking for its exact path. Do not assume it exists in the current local workspace.

## User Preferences / Operating Rules

- User wants aggressive progress and few bounded tests.
- If a method is clearly bad after smoke/correction, close it.
- If a method has real potential, improve it using engineering judgment, not only the original plan.
- Avoid tiny leaderboard-chasing hacks unless they directly answer a question.
- Avoid brittle public-test overfitting. Private set is unknown.
- The user is okay with parallel GPU runs but does not want quality reduced just to fit memory.
- For OOM, first use batch size 1 or DataParallel across GPU 2/3. A100 + A6000 can work but prefer same-memory pair if possible.
- Do not zip on server; user prefers robocopy to local then zip.

## Security Note

The user pasted an HF token earlier in chat. Do not repeat it in files or messages. Use environment variables only. Recommend rotating it if relevant.
