# Track 5 Visual Review Memory

Last updated: 2026-06-06

This document records visual conclusions from images, contact sheets, and
videos reviewed during development. Re-open the referenced server media before
making decisions because this text is a summary, not a replacement for direct
inspection.

## Review Method

For every meaningful validation run:

1. Create contact sheets with 8-12 evenly spaced frames.
2. Inspect the full video when a sheet looks suspicious.
3. Check:
   - scene and background stability;
   - camera motion appropriate for the view type;
   - pedestrian and vehicle displacement;
   - actor deformation, duplication, or disappearance;
   - freeze, sliding, flicker, and long-horizon drift;
   - consistency with the caption.
4. Record findings in this file and `experiment_log.md`.

## Official Wan2.2 Zero-Shot

Main full-test run:

```text
/workspace/experiments/wan_ticft/wan22_ti2v_test_121f_s10
```

Submission contact sheets:

```text
/workspace/experiments/wan_ticft/wan22_ti2v_test_121f_s10_submission_sheets
```

Visual conclusion:

- Images and videos are generally sharp, realistic, and visually attractive.
- Motion is stronger and more natural than the residual-flow refiner.
- Some generated futures are inconsistent with the actual traffic situation.
- Wan may invent or alter objects, actor trajectories, or scene details.
- This matches the leaderboard result: strong CLIP/FVD but weak PSNR/SSIM.
- Keep this as the primary baseline and a candidate source for reranking.

## Residual-Flow Refiner

Main full inference:

```text
/workspace/experiments/refiner_e2e_mixed_h1_16
```

Motion diagnosis showed:

- BDD predicted frame difference approximately `0.324`;
- fixed-camera predicted frame difference approximately `0.018`;
- vehicle/IP predicted frame difference approximately `0.029`;
- freeze rate was effectively 100%.

Visual conclusion:

- Pixel fidelity is good near the conditioning frame.
- Long outputs are nearly frozen.
- Rollout increased motion but caused severe drift.
- Anchor rollout with alpha `0.35` returned to near-freeze.
- Do not use this as the only final generator.

## DiffSynth LoRA 13 Frames

BDD validation:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_smoke_13f_val4
```

Vehicle validation:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_smoke_13f_vehicle4
```

Visual conclusion:

- Format and inference worked.
- Videos preserved the scene but were very conservative.
- Horizon was too short to evaluate the final task.

## DiffSynth LoRA 49 Frames

BDD sheets:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_300_49f_val4_sheets
```

Vehicle sheets:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_300_49f_vehicle4_sheets
```

BDD sample findings:

- `val_bdd_pc_5k_video7040_phase0`: continuity was stable; motion existed but
  was weak.
- `val_bdd_pc_5k_video2149_phase0`: mild forward camera motion looked
  reasonable.
- `val_bdd_pc_5k_video1407_phase0`: scene stayed stable but motion was weak.
- `val_bdd_pc_5k_video3237_phase0`: nearly frozen; pedestrian and open vehicle
  door changed very little through frame 48.

Vehicle sample findings:

- Motion was generally better than BDD.
- Camera advanced and cyclists/vehicles changed position.
- One black vehicle visibly moved across the scene but deformed slightly.
- Overall it hallucinated less than zero-shot, but was too conservative.

Root cause discovered later:

- The original manifest contained only 9 target frames.
- Preparing 49-frame clips repeated the final target frame for the remaining
  duration.
- The LoRA therefore learned a freeze-heavy motion distribution.
- Do not use this checkpoint for final generation.

## Bad 121-Frame Inference From 49-Frame LoRA

Reviewed video:

```text
000000_test_wts_test_wts_overhead_or_fixed_camera_20230707_14_CN16_T1_Camera2_3_phase3.mp4
```

Sample ID:

```text
test_wts_test_wts_overhead_or_fixed_camera_20230707_14_CN16_T1_Camera2_3_phase3
```

Visual conclusion:

- File was technically valid: 121 frames, 15 FPS, `832x480`.
- The pedestrian remained near the same crosswalk location for almost the
  entire video.
- Motion looked like slight sliding/pose changes instead of a real crossing.
- This contradicted the caption, which described crossing at a slightly higher
  speed.
- The main cause was training-horizon/data mismatch, not video encoding.

## DiffSynth 121-Frame Smoke LoRA

Output:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_smoke_121f_test1
```

Sheets:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_smoke_121f_test1_sheets
```

Visual conclusion:

- Cleaner and less deformed than the 49-frame LoRA forced to 121 frames.
- Scene stability improved.
- Pedestrian moved slightly more across the full horizon.
- Motion was still too conservative and did not match the captioned crossing
  speed.
- The smoke model used only 20 clips, so this did not invalidate the 121-frame
  training direction.

## Latest Model Under Review

Balanced 500-sample 121-frame LoRA:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5
```

The 12-sample BDD alpha `1.0` validation has now received direct visual review.
Vehicle/IP, fixed-overhead, and lower-alpha validation remain required.

## Desired Visual Behavior By View

### Fixed overhead camera

- Camera and background must remain static.
- Road geometry, lighting, and parked/static objects should not change.
- Only actors and physically affected regions should move.
- Actor motion must be visible enough to avoid freeze.

### Vehicle/IP camera

- Permit moderate ego-camera and road motion.
- Maintain lane geometry and nearby actor identity.
- Avoid warping moving vehicles or cyclists.

### BDD ego view

- Permit strong forward motion.
- Preserve road layout, vehicle identity, and horizon consistency.
- Penalize invented objects and large scene changes.

## Visual Decision Standard

Do not approve a model for full test inference merely because:

- all rows returned `status=ok`;
- contact sheets are sharp;
- CLIP similarity is high;
- a single sample looks good.

Approve only after mixed-view validation shows:

- meaningful actor/camera motion;
- no systematic freeze;
- lower hallucination than zero-shot;
- stable long-horizon geometry;
- acceptable deformation and flicker.



## Balanced 500-Sample 121-Frame LoRA BDD Review

Output and sheets:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_sheets
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_dense24/sheets
```

All 12 original contact sheets were directly inspected from the server-mounted
workspace.

Per-sample visual findings:

| Sample | Visual finding |
| --- | --- |
| `video7040_phase0` | Severe hallucination: a pedestrian appears and grows toward the camera while the scene changes. |
| `video2149_phase0` | One of the strongest outputs; forward motion and street geometry remain reasonably consistent. |
| `video1407_phase0` | Severe actor hallucination and deformation; pedestrians become implausibly large near the camera. |
| `video3237_phase0` | Effectively frozen, confirming the quantitative diagnosis. |
| `video6180_phase0` | Motion exists, but late frames contain strong vehicle warping and scene drift. |
| `video6846_phase1` | Pedestrian crossing is visible and mostly coherent, but the person grows too large and becomes threateningly close to the camera. |
| `video8125_phase1` | Strong, plausible ego-motion and comparatively stable vehicles; visually one of the better outputs despite the high motion metric. |
| `video2907_phase1` | Mostly frozen camera/background while a crossing vehicle advances; coherent but overly conservative. |
| `video9213_phase1` | Pedestrian movement is visible, but the actor becomes oversized and remains unnaturally centered near the vehicle. |
| `video6_phase1` | Severe drift and object churn; parked and moving vehicles change identity and geometry. |
| `video8139_phase2` | Scene remains stable, but the pedestrian persists and moves unnaturally through the stopped-vehicle scene. |
| `video8277_phase2` | Background and vehicles are stable, but a new pedestrian appears late and grows toward the camera. |

Quantitative motion review:

| Sample | Adjacent abs | First-last abs | Flow mean | Flag |
| --- | ---: | ---: | ---: | --- |
| `video7040_phase0` | 6.318 | 42.300 | 0.888 | motion |
| `video2149_phase0` | 5.240 | 29.792 | 0.660 | motion |
| `video1407_phase0` | 8.537 | 32.457 | 2.136 | strong motion |
| `video3237_phase0` | 0.459 | 4.082 | 0.020 | near-freeze |
| `video6180_phase0` | 7.371 | 43.556 | 1.286 | strong motion |
| `video6846_phase1` | 3.252 | 30.581 | 0.516 | motion |
| `video8125_phase1` | 11.595 | 41.995 | 1.999 | excessive-motion candidate |
| `video2907_phase1` | 3.106 | 33.847 | 0.296 | motion |
| `video9213_phase1` | 3.543 | 20.884 | 0.672 | motion |
| `video6_phase1` | 11.523 | 52.613 | 3.164 | excessive drift candidate |
| `video8139_phase2` | 3.248 | 25.361 | 0.430 | inspect stopped-vehicle consistency |
| `video8277_phase2` | 4.475 | 40.348 | 0.646 | inspect stopped-vehicle consistency |

Comparison with documented official Wan zero-shot:

- Official Wan generally has strong, natural motion but may hallucinate scene
  details, objects, and trajectories.
- Balanced LoRA alpha `1.0` is not systematically frozen, unlike the residual
  refiner, but one BDD sample is clearly near-frozen and two show excessive
  motion/drift.
- Direct visual evidence shows that alpha `1.0` does not reliably reduce
  hallucination. It frequently invents or enlarges pedestrians and sometimes
  deforms or changes vehicle identities.
- A minority of samples are visually strong, so the checkpoint may still be
  useful at lower alpha or as one candidate in an ensemble.

Decision:

Do not approve alpha `1.0` for full test inference. The failure rate is too
high: at least six of twelve BDD sheets show severe hallucination, deformation,
freeze, or drift. Continue only with vehicle/IP and fixed-overhead validation
plus alpha `0.5` or `0.7` comparison.

## Balanced 121f LoRA BDD Alpha Comparison

Compared runs:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_a05
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_a07
```

All alpha `0.5` and `0.7` outputs completed successfully and all 24 contact
sheets were directly inspected.

Cross-alpha findings:

- `video7040`, `video1407`, `video6180`, and `video6` remain severely broken
  across all tested alpha values.
- `video3237` remains effectively frozen across all tested alpha values.
- `video8125` and `video2907` remain among the more coherent outputs.
- Alpha `0.5` improves the late phase-2 behavior of `video8139` and
  `video8277`, but introduces a large approaching pedestrian in `video2149`.
- Alpha `0.7` keeps `video2149` clean, but its overall failure pattern is close
  to alpha `1.0`.

Conclusion:

Neither alpha `0.5` nor `0.7` provides a meaningful overall improvement over
alpha `1.0`. Lower alpha changes which samples fail, but does not remove the
systematic hallucination, deformation, drift, and freeze modes. Since the same
samples often fail similarly across alpha values, the dominant issue is likely
the generated trajectory/seed and DiffSynth TI2V behavior on BDD rather than
only excessive LoRA strength.

Do not run a full test set using any single one of these DiffSynth LoRA alpha
settings. Continue mixed-view validation, then prioritize multi-seed candidate
generation and reranking over further alpha-only tuning.

## Balanced 121f LoRA WTS Alpha 0.7 Review

Outputs:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_vehicle4_a07
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_fixed4_a07
```

All eight contact sheets were directly inspected.

Vehicle/IP findings:

- `20230707_18_CY23`: coherent road continuation with restrained motion.
- `20230922_9_SN13`: severe hallucination; a truck appears and a pedestrian
  grows implausibly large near the camera.
- `20230922_31_CN33`: scene remains mostly coherent, but motion is weak and
  actor behavior is uncertain.
- `20230922_43_SN36`: near-freeze.

Vehicle/IP decision:

Do not use this LoRA as the primary vehicle/IP generator. The view still shows
the same hallucination and freeze failure modes seen on BDD.

Fixed-overhead findings:

- Background and camera geometry remain exceptionally stable in all four
  samples.
- `20230929_9_SY18`: strongest sample; a white van moves coherently while the
  background remains fixed.
- `20230714_44_CY8`: stable but actor motion is limited.
- `20230922_19_CN13`: nearly frozen.
- `20230929_56_SY20`: mostly static actors with limited late vehicle motion.

Fixed-overhead decision:

Keep the LoRA only as a fixed-overhead pixel-preserving candidate. Its stable
background may help PSNR/SSIM, but the visible freeze risk means it must be
compared quantitatively against repeat-last and must not be used blindly.

## Balanced 121f LoRA WTS Phases 1-4 Review

Outputs:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_fixed_p1to4_a07
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_vehicle_p1to4_a07
```

Both runs completed 20/20 successfully. All 40 contact sheets were directly
inspected.

Fixed-overhead phases 1-4:

- Camera and background geometry remain highly stable across almost every
  sample.
- Many phase-1 and phase-2 outputs are effectively frozen or move actors only
  slightly.
- Later phases more often generate moving vehicles and pedestrians.
- Several later-phase vehicles appear only near the end, grow too large, or
  deform while approaching the camera.
- The branch is useful only as a pixel-preserving candidate and is often close
  to repeat-last behavior.

Vehicle/IP phases 1-4:

- Freeze and weak motion are common.
- Pedestrians frequently remain fixed, disappear, or grow implausibly large.
- Trucks and cars sometimes appear late and expand toward the camera.
- A few road-continuation samples are coherent, but the failure rate remains
  too high for either primary generation or blind ensemble inclusion.

Final LoRA decision:

- Reject DiffSynth LoRA as the primary generator for every view.
- Reject it as a vehicle/IP or BDD ensemble candidate unless an automatic
  reranker explicitly proves that a candidate is safe.
- Retain it only as an optional fixed-overhead candidate.
- Do not spend more compute on LoRA alpha sweeps or broad LoRA inference.
