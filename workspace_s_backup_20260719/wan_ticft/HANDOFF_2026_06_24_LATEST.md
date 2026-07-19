# AI City Track 5 Latest Handoff - 2026-06-24

## Current Goal

We are trying to beat the current best public score around **69.2456** and look for a route that can plausibly move toward **75+**, not tiny postprocess changes.

The current best candidate remains:

```text
/workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate
```

Public result:

```text
depth_layer    General Success 69.2456 18.5942 0.6222 0.3312 0.9397 47.8320 31.2770
```

The user has already copied/synced this folder to:

```text
G:\hybrid_depth_layered_a75_wts_overhead_object_layer_candidate
```

Robocopy dry-run with `/L` showed no files pending, so the `G:` folder is synced. The folder modified time being `21/06` is expected because robocopy preserves source timestamps; created time on `G:` may be later.

## Latest Experiments Since Previous Handoff

### 1. Motion-Persistence Router

Purpose: use `repeat_last` only for clips that are almost static, because repeat can massively improve pixel metrics on static clips but usually destroys FVD.

Validation command/script:

```text
/workspace/wan_ticft/evaluation/run_bdd_motion_persistence_router_gate24.sh
```

BDD24 result:

```text
depth_a75      N=23 PSNR 18.5623 SSIM 0.8344 AbsDiff 19.0541
repeat_video   N=23 PSNR 19.9950 SSIM 0.8083 AbsDiff 17.1749
route_h4_p1    N=23 PSNR 19.0954 SSIM 0.8372 AbsDiff 18.5785
```

Proxy:

```text
depth_a75      FID 5.8550 FVD 19.2549
route_h4_p1    FID 5.7812 FVD 19.0315
repeat_video   FID 6.1906 FVD 25.6481
```

Diagnostics:

```text
/workspace/experiments/wan_ticft/bdd_motion_persistence_router_bdd24/diagnostics.csv
```

Important detail:

```text
route_h4_p1 routed only 1/24 val sample to repeat:
val_bdd_pc_5k_video8277_phase2
depth: PSNR 19.9599 SSIM 0.9299 AbsDiff 13.9279
route: PSNR 32.2211 SSIM 0.9951 AbsDiff 2.9894
```

So the route was locally promising but very conservative.

### 2. Motion-Persistence Test Candidate

Build script:

```text
/workspace/wan_ticft/evaluation/build_motion_persistence_h4p1_bdd_wts_candidate.sh
```

Output:

```text
/workspace/experiments/hybrid_bdd_motion_persistence_h4p1_wts_overhead_object_candidate
```

Verification:

```text
rows=71
frames_found=5194
by_status={'ok': 71}
```

But test diagnostics showed:

```text
routes {'depth': 44}
```

Meaning no BDD test samples routed to repeat. Therefore this candidate is effectively the same as the current depth_a75 best and is **not a meaningful new submission**.

Do not submit this as a distinct candidate unless there is some external reason. It is basically identical to:

```text
/workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate
```

### 3. LTX 13B BF16 Base Predictor

Reason for trying: postprocess/refiners are not moving score enough. A stronger **base video predictor** is one of the few remaining plausible high-upside directions.

Scripts/config prepared:

```text
/workspace/wan_ticft/evaluation/ltxv_13b_dev_local_gate.yaml
/workspace/wan_ticft/evaluation/run_ltx13b_base_bdd4_gate.sh
```

This uses the existing BF16/dev weight:

```text
/workspace/models/ltx13b098_ic_detailer/ltxv-13b-0.9.8-dev.safetensors
```

Full-res BF16 (`704x1216x121`) OOMed during denoise, using about 46GB and needing another ~1.37GB.

Low-res BF16 (`640x1088x121`) passed denoise but OOMed at VAE decode:

```text
Tried to allocate 2.11 GiB
GPU had 1.41 GiB free
Process used 45.98 GiB
```

A forced-offload patch was tried. It exposed that LTX's default code disables offload on GPUs >=30GB:

```python
offload_to_cpu = config.offload_to_cpu and get_total_gpu_memory() < 30
```

Patch added env support in:

```text
/workspace/third_party/LTX-Video/ltx_video/inference.py
```

```python
elif os.environ.get("FORCE_LTX_OFFLOAD", "").lower() in {"1", "true", "yes"}:
    offload_to_cpu = bool(config.offload_to_cpu)
```

However, enabling `FORCE_LTX_OFFLOAD=1` caused a device mismatch in LTX multi-scale:

```text
Expected all tensors to be on the same device, but found cpu and cuda:0
```

So **do not use forced offload** for LTX 13B multi-scale unless someone properly fixes device placement. The BF16 wrapper currently no longer sets `FORCE_LTX_OFFLOAD`.

Parallel/model-parallel note:

```text
CUDA_VISIBLE_DEVICES=2,3 does not automatically split LTX 13B across GPUs.
The current LTX pipeline loads to logical cuda:0. Real model-parallel would require deeper pipeline/device-map changes.
```

### 4. LTX 13B FP8 Base Predictor

This is the recommended next run. It keeps the same 13B base idea but uses FP8 to reduce VRAM, hopefully allowing full-res `704x1216x121` without over-hacking resolution.

Scripts/config prepared:

```text
/workspace/wan_ticft/evaluation/ltxv_13b_dev_fp8_local_gate.yaml
/workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

It will use/download:

```text
/workspace/models/ltx13b098_ic_detailer/ltxv-13b-0.9.8-dev-fp8.safetensors
/workspace/models/ltx13b098_ic_detailer/ltxv-spatial-upscaler-0.9.8.safetensors
```

The spatial upscaler already exists:

```text
/workspace/models/ltx13b098_ic_detailer/ltxv-spatial-upscaler-0.9.8.safetensors
```

Next command to run:

```bash
cd /workspace
conda activate wan22

CUDA_VISIBLE_DEVICES=2 \
bash /workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

Default:

```text
LIMIT=1
HEIGHT=704
WIDTH=1216
NUM_FRAMES=121
```

If full-res FP8 OOMs:

```bash
HEIGHT=640 WIDTH=1088 CUDA_VISIBLE_DEVICES=2 \
bash /workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

If the first FP8 sample passes visual/proxy gate, run BDD4:

```bash
LIMIT=4 CUDA_VISIBLE_DEVICES=2 \
bash /workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

Outputs:

```text
/workspace/experiments/wan_ticft/ltx13b_fp8_base_bdd4_gate_eval.csv
/workspace/experiments/wan_ticft/ltx13b_fp8_base_bdd4_gate_proxy.csv
/workspace/experiments/wan_ticft/ltx13b_fp8_base_bdd4_gate_compare_sheets
```

Gate:

```text
Continue only if ltx13b_fp8 has sane road/actor geometry and improves FID/FVD or pixel metrics vs ltx2b_base/depth_a75.
If it changes objects, road layout, or loses badly on pixel metrics, close 13B base.
```

## Recently Added/Modified Files

Motion persistence:

```text
/workspace/wan_ticft/evaluation/materialize_motion_persistence_router.py
/workspace/wan_ticft/evaluation/run_bdd_motion_persistence_router_gate24.sh
/workspace/wan_ticft/evaluation/materialize_motion_persistence_router_test.py
/workspace/wan_ticft/evaluation/build_motion_persistence_h4p1_bdd_wts_candidate.sh
```

LTX 13B:

```text
/workspace/wan_ticft/evaluation/ltxv_13b_dev_local_gate.yaml
/workspace/wan_ticft/evaluation/run_ltx13b_base_bdd4_gate.sh
/workspace/wan_ticft/evaluation/ltxv_13b_dev_fp8_local_gate.yaml
/workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

Patched:

```text
/workspace/wan_ticft/evaluation/run_ltxv_val_gate.py
```

Patch added optional CLI args:

```text
--height
--width
--num_frames
```

Default remains:

```text
704 x 1216 x 121
```

Third-party local patch:

```text
/workspace/third_party/LTX-Video/ltx_video/inference.py
```

It now supports `FORCE_LTX_OFFLOAD`, but do not set this env for current 13B multi-scale runs because it caused CPU/CUDA device mismatch.

## Practical Next Steps

1. Run LTX 13B FP8 full-res one-sample gate:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

2. Inspect:

```text
/workspace/experiments/wan_ticft/ltx13b_fp8_base_bdd4_gate_compare_sheets
```

3. If promising, run:

```bash
LIMIT=4 CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_ltx13b_fp8_base_bdd4_gate.sh
```

4. If BDD4 passes, next likely step is to render depth-layered over LTX13B FP8 base and score against BDD24 or a small BDD gate. Do not build submission directly from one sample.

5. If FP8 still OOMs or visually fails, close 13B base. At that point most postprocess/refiner routes have failed to beat the 69.2456 best meaningfully.

## Do Not Reopen Without New Evidence

- SkyReels texture transfer
- Cosmos 2.5 guardrail/setup unless someone fixes it cleanly
- LTX 13B IC-LoRA detailer residual blends
- ProPainter clean-anchor actor depth
- train-only retrieval candidate
- supervised residual gate v1
- supervised repeat/depth blend v2/v3
- motion-persistence h4/p1 test candidate as a submission, because test routed 0/44 BDD samples

## Reminder

Do not zip on server. Build and verify folder only; user robocopies to local `G:` and zips there.

