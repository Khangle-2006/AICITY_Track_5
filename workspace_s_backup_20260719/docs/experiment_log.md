# Track 5 Experiment Log

Append new experiments here. Do not delete old entries.

Use this template:

```text
## YYYY-MM-DD - Experiment Name

Goal:

Command/config:

Output directory:

Result:

Visual findings:

Decision:

Next action:
```

## 2026-06-06 - Balanced 500-Sample 121-Frame DiffSynth LoRA

Goal:

Train a LoRA at the actual maximum submission horizon instead of padding a
short target sequence.

Configuration:

- 500 balanced samples grouped by dataset part, view type, and phase.
- 121-frame clips.
- Resolution `832x480`.
- Learning rate `5e-5`.
- LoRA rank 32.
- One epoch.

Output directory:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5
```

Result:

Training completed `500/500` in `1:35:33` without OOM.

Decision:

Inspect checkpoints and validate BDD, vehicle/IP, and fixed-overhead samples
before any full test generation.



## 2026-06-06 - Balanced 121f LoRA Alpha 1.0 BDD Diagnostics

Goal:

Visually and quantitatively review all 12 existing BDD validation videos.

Output directory:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12
```

Result:

- 12/12 videos are valid 121-frame `832x480` outputs.
- Motion diagnostics written to `motion_diagnostics.csv`.
- Mean adjacent-frame absolute difference: `5.722`.
- Mean first-to-last absolute difference: `33.151`.
- One near-freeze: `val_bdd_pc_5k_video3237_phase0`.
- Two excessive-motion/drift candidates: `val_bdd_pc_5k_video8125_phase1`
  and `val_bdd_pc_5k_video6_phase1`.
- Dense 24-frame sheets created for the three flagged samples and both
  phase-2 samples.

Visual findings:

All 12 original sheets were directly inspected through the mounted server
workspace.

- Stronger outputs: `video2149_phase0`, `video8125_phase1`.
- Clear near-freeze: `video3237_phase0`.
- Severe pedestrian hallucination/enlargement: `video7040_phase0`,
  `video1407_phase0`, `video6846_phase1`, `video9213_phase1`,
  `video8277_phase2`.
- Severe vehicle/scene deformation or drift: `video6180_phase0`,
  `video6_phase1`.
- `video2907_phase1` is coherent but conservative.
- `video8139_phase2` is stable but has unnatural pedestrian persistence.

Decision:

Do not run full test inference with alpha `1.0`. It avoids systematic freeze,
but its severe BDD visual failure rate is too high. Continue only as a candidate
through mixed-view validation and lower-alpha comparison.

Next action:

Run four vehicle/IP and four fixed-overhead samples at alpha `1.0`, then repeat
the same rows at alpha `0.5` or `0.7`. Compare the exact same rows and seeds.

## 2026-06-06 - Balanced 121f LoRA BDD Alpha 0.5 And 0.7

Goal:

Determine whether reducing LoRA strength fixes the alpha `1.0` BDD
hallucination and drift failures.

Output directories:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_a05
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_a07
```

Result:

- Alpha `0.5`: 12/12 successful.
- Alpha `0.7`: 12/12 successful.
- All 24 contact sheets directly reviewed.

Visual findings:

- Severe failures persist across all alphas on `video7040`, `video1407`,
  `video6180`, and `video6`.
- `video3237` remains frozen.
- `video8125` and `video2907` remain comparatively coherent.
- Alpha `0.5` improves some phase-2 samples but introduces a large approaching
  pedestrian in `video2149`.
- Alpha `0.7` behaves similarly to alpha `1.0`.

Decision:

Lowering alpha alone is insufficient. Do not run full-test DiffSynth inference
with any tested single-alpha setting.

Next action:

Validate vehicle/IP and fixed-overhead samples. For BDD, prioritize multi-seed
candidate generation and reranking rather than additional alpha-only sweeps.

## 2026-06-06 - Balanced 121f LoRA WTS Alpha 0.7

Goal:

Determine whether the balanced LoRA is more useful on vehicle/IP or
fixed-overhead views than on BDD.

Outputs:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_vehicle4_a07
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_fixed4_a07
```

Result:

- Vehicle/IP: 4/4 successful.
- Fixed-overhead: 4/4 successful.
- All eight contact sheets directly inspected.

Visual findings:

- Vehicle/IP still exhibits severe hallucination and near-freeze failures.
- Fixed-overhead backgrounds and camera geometry are highly stable.
- Fixed-overhead actor motion is often too weak; one sample is nearly frozen.

Decision:

- Reject the LoRA as the primary BDD or vehicle/IP generator.
- Retain it only as a fixed-overhead candidate pending quantitative validation
  against repeat-last.
- Stop alpha-only sweeps.

Next action:

Quantitatively compare repeat-last and LoRA on a larger fixed-overhead
validation subset. Begin implementing official-Wan multi-seed generation and
view-specific reranking for BDD and vehicle/IP.

## 2026-06-06 - WTS Phases 1-4 Expanded LoRA Review

Goal:

Validate whether the LoRA remains useful outside phase 0.

Result:

- Fixed-overhead: 20/20 successful, five samples for each phase 1-4.
- Vehicle/IP: 20/20 successful, five samples for each phase 1-4.
- All 40 contact sheets directly reviewed.

Visual findings:

- Fixed-overhead backgrounds remain stable, but freeze is common. Later phases
  sometimes contain useful motion and sometimes hallucinated late vehicles.
- Vehicle/IP contains frequent freeze, weak motion, oversized actors, and
  late-appearing vehicles.

Decision:

End broad DiffSynth LoRA experimentation. Retain LoRA only as an optional
fixed-overhead candidate. Move primary effort to history-aware multi-seed
generation and reranking.

## 2026-06-07 - Official Wan BDD Multi-Seed Initial Evaluation

Goal:

Measure whether multiple official Wan seeds improve short-horizon BDD
validation fidelity.

Candidates:

- official Wan seed 101;
- official Wan seed 1001;
- DiffSynth LoRA alpha 0.7;
- repeat-last.

Initial result:

- Repeat-last won 9/12 samples.
- LoRA won 2/12.
- Wan seed 101 won 1/12.
- Wan seed 1001 won 0/12.

Evaluator issue discovered:

The first evaluation used source-video frame offsets directly. Source
validation videos are approximately 30 FPS, official Wan outputs are 24 FPS,
and DiffSynth outputs are 15 FPS. This sampled generated candidates at the
wrong timestamps. `wan_ticft/evaluation/evaluate_video_reports.py` was updated
to map target timestamps using source and generated video FPS.

Next action:

Re-run only the evaluator using the existing generated videos. Do not regenerate
the candidates.

Timestamp-corrected result:

- Repeat-last: `21.1324` PSNR, `0.8457` SSIM, wins 10/12.
- LoRA alpha 0.7: `19.2688` PSNR, `0.8320` SSIM, wins 1/12.
- Wan seed 101: `18.0704` PSNR, `0.8184` SSIM, wins 1/12.
- Wan seed 1001: `18.1364` PSNR, `0.8187` SSIM, wins 0/12.
- Oracle per-sample candidate selection: `21.2418` PSNR and `0.8491` SSIM.
- Oracle improves over repeat-last by only approximately `+0.1094` PSNR and
  `+0.0035` SSIM.

Interpretation:

The current validation manifest has only nine target frames spanning about 20
source frames, approximately 0.67 seconds. It strongly favors repeat-last and
does not adequately measure the long forecasting horizon. Build a real
121-frame validation manifest and re-evaluate the existing generated videos
before deciding whether multi-seed Wan is useful.

## 2026-06-07 - Matched 121-Frame BDD Candidate Evaluation

The first 12 BDD validation rows were materialized with 120 consecutive real
future frames after the final condition frame:

```text
/workspace/experiments/wan_ticft/manifests/val_ticft_bdd12_matched_121f_v2.jsonl
```

Verification:

- 12 rows;
- 48 condition files;
- 1440 target files;
- zero bad or zero-byte files.

Results:

- repeat-last: `15.8025` PSNR, `0.6718` SSIM, wins 10/12;
- LoRA alpha 0.7: `14.6321` PSNR, `0.6381` SSIM, wins 1/12;
- official Wan seed 1001: `14.6018` PSNR, `0.6501` SSIM, wins 1/12;
- official Wan seed 101: `14.0731` PSNR, `0.6162` SSIM, wins 0/12.

The ground-truth oracle across these candidates reaches only approximately
`15.8737` PSNR, about `+0.0712` over repeat-last. Therefore, adding more seeds
or building a reranker over the current candidate family is unlikely to
materially improve pixel fidelity.

Next action:

Evaluate time-normalized optical-flow/history-aware baselines on the same
matched 121-frame manifest. The existing four condition frames are separated
by multiple source frames, so flow velocity must be normalized by the final
condition-frame gap.

## 2026-06-07 - Matched 121-Frame BDD Motion Baselines

Time-normalized optical-flow baselines were evaluated on the same 12 BDD rows.

Results:

- repeat-last: `15.8025` PSNR, `0.6718` SSIM;
- flow-normalized light: `16.0701` PSNR, `0.6875` SSIM;
- flow-normalized medium: `16.2702` PSNR, `0.6994` SSIM;
- flow-normalized long: `16.1887` PSNR, `0.6942` SSIM;
- region-normalized medium: `16.1848` PSNR, `0.6985` SSIM.

The best single configuration, flow-normalized medium, improves repeat-last by
approximately `+0.4677` PSNR and `+0.0276` SSIM. The oracle across all five
motion candidates reaches only `16.3068` PSNR, about `+0.0366` above the best
single configuration, so a motion-candidate reranker is not currently a
priority.

Next action:

Evaluate the same motion candidates on all 125 balanced validation rows and all
three camera-view distributions. Decode 120 consecutive future frames directly
from source videos to avoid materializing another large PNG dataset.

Initial all-125 evaluation caveat:

The first direct-source run used OpenCV sequential decoding. It produced
invalid BDD alignment: the same matched-12 BDD rows dropped from approximately
`15.8` repeat-last PSNR to approximately `10.0`, with negative SSIM. This
contradicts the FFmpeg-materialized matched evaluation. The initial all-125
output must not be used for model selection.

The evaluator was changed to use the validated FFmpeg extractor in temporary
directories. Re-run BDD first, then all views only after the matched BDD
metrics are reproduced.

BDD FFmpeg-aligned 120-frame stress-test result:

- 24/25 BDD samples were valid;
- flow-normalized medium was the best mean-PSNR configuration at `16.3392`
  PSNR and `0.6369` SSIM;
- repeat-last achieved `16.1637` PSNR and `0.6129` SSIM.

The forced 120-frame evaluation is still only a stress test. The authoritative
next comparison uses every row's existing verified target PNGs and predicts at
the exact source-frame offsets recorded in the manifest. This includes all
views and avoids excluding short phase segments.

## 2026-06-07 - All-125 Sparse-Target Motion Evaluation

All 125 balanced validation rows were evaluated at their exact verified target
frame offsets.

Overall:

- repeat-last: `25.5895` PSNR, `0.9070` SSIM;
- flow-normalized long: `26.0207` PSNR, `0.9176` SSIM;
- region-normalized medium: `26.7591` PSNR, `0.9275` SSIM.

Region-normalized medium improves repeat-last by approximately `+1.1696` PSNR
and `+0.0205` SSIM. It wins 88/125 samples and is best on average for BDD,
fixed-overhead, vehicle/IP, and every event phase. The oracle across all five
motion candidates reaches `26.8511` PSNR, only approximately `+0.0920` above
region-normalized medium.

Decision:

- Promote region-normalized medium as the primary pixel-fidelity candidate.
- Implement it exactly in the test generator; do not substitute the older
  `region_flow_blend`, which uses different scaling and horizon-gain behavior.
- Run a four-sample test smoke and verify output structure before generating
  the full 71-sample flat submission archive.

Test smoke result:

- 4/4 BDD samples generated and verified;
- 256 total frames with exact requested counts;
- all four contact sheets and montages were directly reviewed;
- scenes and cameras remain stable with no severe hallucination or deformation;
- motion is intentionally restrained and saturates over the horizon.

Before full generation, run a small mixed-view smoke because the first sorted
four test rows are all BDD and the regional mask strength is view-specific.

Mixed-view smoke result:

- 2/2 BDD, 2/2 fixed-overhead, and 2/2 vehicle/IP samples generated;
- all six contact sheets and montages were directly reviewed;
- no severe ghosting, road deformation, scene change, or actor hallucination;
- fixed-overhead masks remain appropriately small, approximately `0.07` mean
  mask strength in the reviewed samples;
- actor and ego motion remain restrained and pixel-preserving.

Proceed with the full 71-sample region-normalized generation. Generate and
review contact sheets before creating the final ZIP archive.

Full test generation result:

- 71/71 samples successful;
- exactly 5194 requested frames;
- 44 BDD, 7 fixed-overhead, and 20 vehicle/IP samples;
- submission verification passed for every sample;
- 71/71 contact sheets created.

Visual QA:

- directly reviewed the mixed-view smoke plus the highest-mask and
  highest-motion full-test outliers;
- fixed-overhead and vehicle/IP outputs preserve background geometry and show
  no severe actor hallucination;
- high-motion BDD outputs contain occasional mild warp/ghost artifacts but no
  failure severe enough to justify replacing them with repeat-last;
- motion remains restrained and saturates, consistent with the validated
  pixel-fidelity objective.

Decision:

Package region-normalized medium as a separate submission candidate. Do not
overwrite or discard the official Wan baseline because the region-normalized
candidate is expected to improve PSNR/SSIM while potentially reducing
semantic/generative metrics such as CLIP, FID, and FVD.

Packaged submission:

```text
/workspace/experiments/region_norm_medium_submission_flat.zip
```

- archive size: approximately 6.12 GB;
- 71/71 samples verified;
- exactly 5194 PNG frames;
- per-sample frame counts range from 51 to 120;
- flat archive layout with sample folders directly at the ZIP root.

## 2026-06-14 - LTX Alignment And Color Audit

Goal:

Determine whether the low BDD LTX pixel fidelity is primarily caused by a
systematic temporal, spatial, channel-order, or global color error.

Result:

- Baseline LTX: `16.9166` PSNR, `0.7547` SSIM, `24.2222` AbsDiff.
- Best PSNR variant was cross-fitted RGB affine correction at `17.3059`
  PSNR, a gain of only `+0.3893`.
- RGB affine reduced SSIM to `0.7347` and increased AbsDiff to `24.7575`.
- Best temporal shift was `-2` at `16.9909` PSNR, only `+0.0743`.
- Spatial shifts and BGR channel order did not improve the result.

Decision:

Gate failed the predefined `+0.5 dB` threshold. Do not modify the submission
pipeline with global color correction or fixed alignment shifts.

Next action:

Run the bounded oracle future-warp plus LTX hole-fill gate. Continue toward a
warp-and-inpaint system only if hard-compositing LTX into oracle splat holes
clearly beats both LTX transition16 and the forward splat baseline.

## 2026-06-14 - BDD Oracle Warp And LTX Hole Fill

Goal:

Measure whether pixel-space future warp plus generative filling only in
disocclusion holes has enough headroom to justify further work.

Result:

- LTX transition16: `17.6807` PSNR, `0.7699` SSIM.
- Oracle backward flow: `21.3615` PSNR, `0.8566` SSIM.
- At dilation 4, LTX hole fill reached `21.1874` PSNR versus `20.7301` for
  repeat-last hole fill: `+0.4573` PSNR and `+0.0200` SSIM.
- At dilation 8, LTX hole fill improved repeat fill by `+0.6194` PSNR and
  `+0.0278` SSIM.
- Target hole fill at dilation 4 reached `22.4536` PSNR, leaving `+1.7234`
  PSNR headroom over repeat fill.
- Dilation-4 and dilation-8 masks covered `12.43%` and `19.60%` respectively.

Decision:

The BDD QW2 oracle gate passed. Do not test ProPainter yet: LTX hole filling
already passed, and the dominant gain still comes from oracle future flow.
Before any inpainting/model expansion, test whether warp and disocclusion
headroom exists across BDD, WTS fixed, and WTS vehicle/IP.

Next action:

Run a six-sample cross-domain oracle smoke using one shared dilation setting
family. Stop the general warp-and-inpaint direction if any domain lacks useful
hole-filling headroom.

## 2026-06-14 - Cross-Domain Warp And Hole Oracle Smoke

Goal:

Check whether the BDD warp-and-disocclusion conclusion also holds on WTS fixed
and WTS vehicle/IP before expanding the direction.

Result:

- BDD dilation-4 target fill beat repeat fill by `+2.0057` PSNR with a
  `12.09%` mask.
- WTS fixed dilation-4 target fill beat repeat fill by `+0.7137` PSNR with a
  `0.28%` mask.
- WTS vehicle/IP dilation-4 target fill beat repeat fill by `+0.9181` PSNR
  with a `4.23%` mask.
- All three domains passed the `+0.50` PSNR and `<25%` mask smoke thresholds.

Decision:

The mechanism has cross-domain oracle headroom, but two samples per domain are
not sufficient for promotion. Run one final bounded oracle gate with ten
samples per domain. Do not run ProPainter, train a model, or evaluate all 125
rows before this gate passes.

Next action:

Run `run_cross_domain_warp_hole_oracle_gate30.sh`. If every domain passes,
stop oracle work and test one deployable future-flow approach. If any domain
fails, close the general warp-and-inpaint direction.

## 2026-06-14 - Cross-Domain Warp And Hole Oracle Gate 30

Goal:

Confirm the cross-domain oracle mechanism on ten samples per domain.

Result:

- BDD dilation-4 target fill improved repeat fill by `+2.0180` PSNR with a
  `10.40%` mask.
- WTS vehicle/IP improved by `+1.3986` PSNR with a `3.82%` mask.
- WTS fixed improved by only `+0.2953` PSNR with a `0.18%` mask.
- Oracle backward flow improved repeat-last strongly in every domain:
  approximately `+6.38` BDD, `+1.54` WTS fixed, and `+5.64` WTS vehicle/IP
  PSNR.

Decision:

Close the general warp-and-inpaint direction because WTS fixed failed the
predefined `+0.50` hole-filling threshold. Disocclusion filling is not a
general bottleneck; fixed-camera holes are negligible.

Future-motion prediction remains a cross-domain bottleneck because oracle
backward flow has large gains in every domain. The previous learned predictor
used Farneback pseudo-labels and regression loss and failed. Permit exactly
one new feasibility audit for official pretrained MemFlow-P future-flow
prediction. Do not train, download checkpoints, or reopen ProPainter yet.

Next action:

Run `run_memflow_p_feasibility.sh`. Stop if MemFlow-P is not directly usable,
requires substantial environment repair, or supports only an impractical
one-step path for this task.

## 2026-06-14 - MemFlow-P Future-Flow Feasibility

Goal:

Determine whether official pretrained MemFlow-P can provide a deployable
history-to-future-flow path without training or substantial integration work.

Result:

- Official source contains `MemFlowNet_P` and a future-prediction evaluation
  path.
- The documented and implemented prediction task is one time step ahead.
- Memory for subsequent predictions is updated using an observed image pair
  and its estimated or ground-truth flow. Native inference does not provide a
  future-frame-free rollout path for 51-120 steps.
- A long rollout would require feeding synthetic warped predictions back into
  the model, introducing an unsupported autoregressive drift mode.
- The repository targets PyTorch `1.13.1` and CUDA `11.6`, while the current
  environment uses PyTorch `2.5.1` and CUDA `12.1`.
- The prediction inference source also references an undefined `end` variable
  inside `InferenceCore.step`, requiring repair before direct use.

Decision:

Close MemFlow-P. Do not download checkpoints, repair the environment, or build
an autoregressive rollout. It fails the predefined feasibility gate because
only one-step prediction is directly supported and integration risk is high.

Next action:

Return to deployable, history-only approaches that can predict the complete
horizon without recursive synthetic-frame input. Require a cross-domain smoke
and stop after one failed correction.

## 2026-06-14 - Multi-History Robust Flow Proposal

Goal:

Test whether unused history frames improve the deployable
`region_norm_medium` baseline without training, new models, or domain-specific
routing.

Method:

- Compute flow for every consecutive history-frame pair.
- Normalize each flow by its actual source-frame gap.
- Take a component-wise median velocity.
- Convert back to the validated final-history-gap scale and reuse the exact
  `region_norm_medium` mask, decay, and horizon cap.

Gate:

Run two samples per domain and compare only `region_norm_medium` against
`region_multi_history_median`. Pass only if macro PSNR improves by at least
`+0.15`, macro SSIM improves, and no domain loses more than `0.05` PSNR.
Allow only one correction, a recency-weighted median, if the first smoke
fails. Close the direction if the correction also fails.

Initial smoke result:

- Macro PSNR decreased from `25.4676` to `25.2219` (`-0.2457`).
- BDD improved by `+0.1377` PSNR.
- WTS fixed decreased by `-0.0969` PSNR.
- WTS vehicle/IP decreased by `-0.7778` PSNR.

Decision:

The unweighted median failed. Permit the predefined single correction:
component-wise recency-weighted median with exponentially increasing weights
`1.5^k` for newer valid history pairs. This keeps recent flow influential
without collapsing to the final pair. If this correction fails the same gate,
close multi-history flow with no additional weighting or tuning.

Final correction result:

- Recency-weighted median produced the same aggregate results as the
  unweighted median.
- Macro PSNR remained below `region_norm_medium` by `-0.2457`.
- BDD improved by `+0.1377` PSNR, but WTS fixed decreased by `-0.0969` and
  WTS vehicle/IP decreased by `-0.7778`.

Decision:

Close multi-history flow. Do not test additional means, weights, trajectory
fits, or aggregation variants.

## 2026-06-14 - Cross-Domain History Camera-Motion Blend Proposal

Goal:

Test whether the previously BDD-positive global camera-motion blend
generalizes without domain routing.

Method:

- Reuse the exact `region_norm_medium` prediction.
- Estimate one partial-affine global transform from the final observed pair.
- Extrapolate it with the existing bounded four-history-gap horizon.
- Blend the global prediction at the previously tested fixed weight `0.20`.

Prior evidence:

On the existing BDD25 ego-motion benchmark, `region_affine_g20` improved PSNR
from `21.0734` to `21.2492` and SSIM from `0.8479` to `0.8532`.

Gate:

Run two samples per domain. Pass only if macro PSNR improves by at least
`+0.10`, macro SSIM does not decrease, and no domain loses more than `0.05`
PSNR. If it fails, close history-derived global camera blending with no weight
or transform-model sweep.

Smoke result:

- Macro PSNR improved from `25.4676` to `25.7071` (`+0.2395`).
- Macro SSIM improved from `0.8985` to `0.9058` (`+0.0073`).
- BDD improved by `+0.3823` PSNR.
- WTS vehicle/IP improved by `+0.3760` PSNR.
- WTS fixed decreased by `-0.0396` PSNR, within the predefined `-0.05` limit.
- The candidate won `4/6` samples and all rows completed.

Decision:

The smoke passes. Run one unchanged balanced gate with ten samples per domain.
Apply the same pass criteria. Do not sweep the global blend weight or
transform model.

Balanced gate30 result:

- Macro PSNR improved from `27.4300` to `27.5889` (`+0.1589`).
- Macro SSIM improved from `0.9344` to `0.9374` (`+0.0030`).
- BDD improved by `+0.2030` PSNR.
- WTS vehicle/IP improved by `+0.2808` PSNR.
- WTS fixed decreased by only `-0.0073` PSNR.
- The candidate won `21/30` samples and all rows completed.

Decision:

The balanced gate passes. Evaluate the unchanged candidate on all 125 balanced
validation rows. Promote only if macro PSNR improves, macro SSIM does not
decrease, and no domain loses more than `0.05` PSNR. Require visual QA before
building a test candidate.

Full balanced validation result:

- Macro PSNR improved from `26.7591` to `26.8733` (`+0.1142`).
- Macro SSIM improved from `0.9275` to `0.9302` (`+0.0027`).
- BDD improved by `+0.1760` PSNR and won `18/25` samples.
- WTS vehicle/IP improved by `+0.2728` PSNR and won `43/50` samples.
- WTS fixed decreased by `-0.0751` PSNR, SSIM decreased by `-0.0006`, and
  AbsDiff increased by `+0.0374`.

Decision:

The candidate fails the predefined promotion rule because WTS fixed loses more
than `0.05` PSNR. Close history-derived global camera blending. Do not sweep
weights, transform models, or add a domain/view router. Retain
`region_norm_medium` as the general classical baseline.

## 2026-06-14 - Remaining Generative Candidate Audit

The previous DiffSynth LoRA fixed-overhead outputs are no longer present on
disk. Existing direct visual review already found common freeze, weak actor
motion, and occasional late hallucinated vehicles. Close the optional
fixed-overhead LoRA branch rather than spending GPU time recreating weak
outputs for quantitative confirmation.

## 2026-06-14 - LTX History-Length Proposal

Goal:

Test whether the current maximum `41`-frame LTX conditioning history causes
unnecessary motion and lighting drift.

Method:

- Keep the same LTX 2B checkpoint, pipeline, prompt, seed, output length, and
  dense40 evaluator.
- Compare one shorter compatible history length, `17` frames, against the
  existing `41`-frame outputs.
- Use two of the weakest and two of the strongest existing LTX dense40
  samples to expose both recovery potential and regression risk.

Gate:

Pass only if history17 improves macro PSNR by at least `+0.15`, does not reduce
macro SSIM, and no sample loses more than `0.50` PSNR. Require visual review
before expanding. If it fails, close LTX history-length tuning with no length
sweep.

History17 smoke result:

- Macro PSNR decreased from `17.5370` to `16.9732` (`-0.5638`).
- Macro SSIM decreased from `0.7199` to `0.7047` (`-0.0152`).
- History17 improved `video3207` by `+0.0349` PSNR and `video4166` by
  `+0.0994`, but decreased `video6180` by `-0.5093` and `video5447` by
  `-1.8806`.
- Visual review found no general scene collapse. History17 avoided a
  hallucinated trailing vehicle in `video4166`, but lost useful long motion
  context in the strong night-driving sample `video5447`.

Decision:

History17 fails, but the direction has a concrete context-length trade-off and
wins `2/4` samples without systematic visual collapse. Permit one correction:
test the midpoint compatible length `25` on the same four rows. This is not a
length sweep. If history25 fails, retain history41 and close LTX
history-length tuning.

History25 correction result:

- Macro PSNR decreased from `17.5370` to `17.4114` (`-0.1256`).
- Macro SSIM decreased from `0.7199` to `0.7188` (`-0.0011`).
- AbsDiff improved slightly from `28.8463` to `28.7185`.
- History25 again improved `video3207` and `video4166`, while history41 again
  won `video5447` and `video6180`.

Interpretation:

The repeated split suggests an observable maneuver-dependent context trade-off
rather than random length noise. The improved rows describe a stationary
vehicle or a turning maneuver; the degraded rows describe straight driving.
The official LTX interface supports sequence conditioning but does not specify
one universally optimal conditioning length.

Decision:

Run one held-out behavior gate using two previously unseen stationary rows and
two straight-driving controls. Promote only a deployable caption-based policy:
history25 for stationary vehicle captions and history41 otherwise. If both
held-out stationary rows do not improve, close history-length routing.

Adaptive-history held-out result:

- Both stationary rows improved with history25:
  - `video6075`: `+0.0714` PSNR and `+0.0008` SSIM.
  - `video8170`: `+0.6282` PSNR and `+0.0155` SSIM.
- Both straight-driving controls degraded with history25.
- The stationary-group mean improved by approximately `+0.3498` PSNR.

Deployment audit:

The only two strict zero-speed BDD test rows both have `history_K=16`. LTX
uses the same compatible `9` history frames whether the cap is `25` or `41`.
Therefore the validated policy cannot change either eligible test output.

Decision:

Close adaptive history-length routing. Retain the universal cap of `41`.

## 2026-06-14 - LTX Validation Prompt Mismatch

Audit finding:

`run_ltxv_val_gate.py` constructed prompts from top-level
`caption_pedestrian` and `caption_vehicle` fields, but the balanced validation
manifest stores the combined caption only in `prompt`. Existing LTX validation
runs therefore used nearly blank semantic prompts, while test LTX generation
uses actual caption fields from `test_index.csv`.

Decision:

Add an explicit manifest-prompt validation mode and run one four-sample gate
against the existing legacy blank-prompt outputs. This tests a real
validation/submission mismatch rather than another output-space adjustment.

Manifest-prompt smoke result:

- Macro PSNR improved slightly from `17.5370` to `17.5544` (`+0.0174`).
- Macro SSIM improved slightly from `0.7199` to `0.7204` (`+0.0005`).
- AbsDiff improved from `28.8463` to `28.2982`.
- Only `video4166` won by PSNR; the other three rows lost slightly.
- Visual review showed that the long detailed prompt can cause literal actor
  hallucination. In `video5447`, a pedestrian appeared close to the camera
  even though preserving existing geometry was preferable.

Decision:

The prompt direction is mixed but has a specific correctable failure mode.
Permit one correction: a concise behavior-only prompt that keeps vehicle
motion cues and explicitly forbids adding, removing, enlarging, or replacing
actors. If this correction fails, retain the legacy prompt and close LTX
prompt tuning.

Behavior-only prompt result:

- Macro PSNR was effectively unchanged: `17.5370` to `17.5353` (`-0.0017`).
- Macro SSIM improved from `0.7199` to `0.7224` (`+0.0025`).
- AbsDiff worsened from `28.8463` to `29.2012`.
- It improved `video4166` by `+0.3190` PSNR and `video5447` by `+0.4697`,
  while removing the manifest-prompt pedestrian hallucination in the night
  scene.
- It degraded `video6180` by `-0.7659` PSNR and `video3207` slightly.

Decision:

The correction fails because one sample loses more than `0.50` PSNR and the
benefit is not predictable from an observable behavior category. Retain the
legacy prompt behavior and close LTX prompt tuning. Do not build a prompt
router or continue wording experiments.

## 2026-06-14 - Driving World Model Research Pivot

Prompt and history tuning cannot provide the required improvement. Current
official-source review identified Epona as the strongest next feasibility
candidate:

- driving-specific autoregressive diffusion world model;
- accepts historical driving context and generates flexible long horizons;
- provides a custom-input `test_demo.py` path described as self-predicted;
- publishes NuPlan/NuScenes world-model checkpoints and temporal DCAE;
- official documentation states inference can run on one RTX 4090.

Important caveat:

Custom input documentation still requires `pose.npy` and `yaw.npy`. Audit the
source to determine whether only observed history pose is needed and future
trajectory is predicted internally, or whether dataset-specific future pose is
effectively required.

Decision:

Run a source-only Epona feasibility audit. Do not download checkpoints or
install an environment until custom history-to-future inference, resource
requirements, and output contract pass.

Source-only audit result:

- `test_demo.py` accepts custom RGB history and uses `self_pred_traj=True`.
- Only observed-history pose/yaw is required; future pose is predicted and fed
  back autoregressively.
- Official checkpoints total about 6.5 GB for one world model plus temporal
  DCAE, and official documentation states single-RTX-4090 inference.
- The released custom-input path contains bounded integration defects:
  undefined `rot_matrix`, a missing default config path, a pose/yaw shape
  mismatch in the documentation, and a loop that generates six frames even
  when only one is requested.
- Default inference uses 100 trajectory diffusion steps plus 100 visual
  diffusion steps per future frame. Runtime is therefore the main unresolved
  feasibility risk.

Decision:

Repair only the bounded custom-input defects and run one frame at the official
100 sampling steps using the combined NuPlan+nuScenes checkpoint. Prepare ten
observed frames at the trained 5 FPS and use caption-derived ego speed with
zero yaw. Do not expand beyond one frame unless runtime and visual structure
are credible.

One-frame integration progress:

- The combined world model and temporal DCAE load successfully on one GPU.
- History encoding takes about `6.6` seconds. One future frame takes about
  `0.09` seconds MST, `0.41` seconds TrajDiT, and `3.58` seconds VisDiT.
- Total cold-start runner time is `49` seconds.
- The released custom demo path required several bounded repairs before
  inference: repo-root import, broken config parser bypass, locale fallback,
  and moving pose/yaw tensors to CUDA.
- Inference completed, but the evaluator path failed after generation. Visual
  inspection also exposed an orientation/aspect mismatch in this validation
  sample: the decoded source is portrait-oriented while Epona resizes it
  directly to landscape `1024x512`.

Decision:

Finalize metrics from the existing output without rerunning inference. If the
orientation mismatch causes a large loss, permit one orientation-aware input
correction before deciding whether to close Epona.

Native one-frame result:

- Epona: `16.9874` PSNR / `0.4972` SSIM / `23.7977` AbsDiff.
- Repeat-last: `15.9408` PSNR / `0.4447` SSIM / `26.5582` AbsDiff.
- Epona improves the first 5 FPS future frame by `+1.0466` PSNR and `+0.0526`
  SSIM despite direct portrait-to-landscape distortion.
- Model memory after completion is approximately `16.1 GB` on RTX A6000.

Decision:

Epona passes one-frame feasibility and remains open. Run one bounded
orientation-aware correction: rotate portrait source frames counterclockwise
before Epona and rotate the generated landscape frame clockwise back to the
evaluation contract. Continue to a small multi-sample gate only if the
correction still beats repeat-last and improves over native Epona.

Orientation correction result:

- Corrected Epona: `15.8612` PSNR / `0.4681` SSIM / `28.1982` AbsDiff.
- It loses `-0.0796` PSNR to repeat-last and loses `-1.1262` PSNR versus
  native Epona.

Decision:

Close orientation correction and retain native preprocessing. The native
one-frame result remains promising, so run a four-sample one-frame
generalization gate on previously unseen BDD scenes before testing rollout.

Four-sample one-frame result:

- Macro PSNR improves from `17.1851` to `17.5077` (`+0.3226`).
- Macro SSIM improves from `0.6348` to `0.6409` (`+0.0061`).
- Epona wins `3/4` samples; worst loss is only `-0.1613` PSNR.
- Runtime is `77` seconds cold-start for four samples. After the first sample,
  each future frame costs approximately four seconds.

Decision:

The one-frame generalization gate passes, narrowly. Before any larger gate,
test four-frame autoregressive drift on the strongest winning sample and the
night sample that lost. Close Epona if step-four performance falls below
repeat-last or aggregate SSIM degrades.

Four-frame autoregressive rollout result:

- Aggregate delta is `-0.4491` PSNR and `-0.0794` SSIM versus repeat-last.
- Only `4/8` sample-step rows win.
- Step-four mean falls by `-1.1183` PSNR and `-0.1264` SSIM.
- The strong daytime sample decays from `+0.7450` PSNR at step one to
  `-0.0547` at step four.
- The night sample decays from `+0.1187` PSNR at step one to `-2.1820` at
  step four, with SSIM delta reaching `-0.2211`.
- Runtime is approximately four seconds per generated future frame after
  model load.

Decision:

Close autoregressive Epona rollout. The one-frame gain does not survive even
four 5-FPS steps, and one-frame-only routing is not deployable for the required
long horizon. Do not run a balanced Epona gate or tune checkpoint, pose, or
sampling steps. Retain the downloaded model only as a research artifact.

## 2026-06-15 - Epona Keyframe Hybrid Proposal

The failed rollout shows that Epona's autoregressive feedback is the failure
mode, while independently predicted first frames improved one-frame BDD
metrics. Permit one offline hybrid test that does not reopen Epona rollout:

- use the single Epona prediction as a future keyframe at source offset `+6`;
- compute its residual relative to `region_norm_medium` at the same offset;
- ramp the residual into the keyframe and decay it fully back to the stable
  region baseline over the following 12 source frames;
- evaluate 24 consecutive future source frames on the four existing Epona
  gate samples without additional GPU inference.

Pass only if the hybrid improves region by at least `+0.10` macro PSNR,
does not reduce macro SSIM, wins at least `3/4`, and no sample loses more than
`0.50` PSNR. Allow one motion-region-masked residual correction if needed.

Initial full-frame residual result:

- Hybrid loses `-3.9194` macro PSNR and `-0.2794` macro SSIM versus
  `region_norm_medium`.
- It wins `0/4` samples and the worst sample loses `-6.0956` PSNR.

Decision:

The full-frame Epona residual is unusable because global appearance and
geometry differences overwhelm the region baseline. Permit the predefined
single correction: apply the same residual only inside the existing
`region_norm_medium` motion mask. If this correction fails, close all Epona
keyframe hybrid work.

Motion-region-masked correction result:

- Masked hybrid loses `-2.2853` macro PSNR and `-0.1158` macro SSIM versus
  `region_norm_medium`.
- It wins `0/4` samples and the worst sample loses `-3.1643` PSNR.

Decision:

Close all Epona work. Its independently generated first frame can beat
repeat-last, but it does not beat the stronger region baseline over a usable
horizon, autoregressive rollout drifts rapidly, and both full-frame and
motion-masked keyframe hybrids fail strongly. Do not test further masks,
weights, anchors, checkpoints, poses, or sampling-step variants.

## 2026-06-15 - Cosmos Predict 2.5 Source Audit

Cosmos-Predict2.5 base 2B is the next bounded feasibility candidate:

- accepts custom Video2World inputs;
- generates `93` output frames directly at 16 FPS instead of predicting one
  future frame recursively;
- provides a standalone Diffusers inference script with configurable output
  frames, inference steps, and `device_map`;
- native inference supports offloading diffusion, text encoder, tokenizer,
  and guardrail models;
- the host driver satisfies the documented CUDA 12.8 minimum.

The specialized autonomous-driving multiview model is rejected because it
requires seven camera inputs and at least eight 80GB GPUs. Audit only the base
2B post-trained Video2World model.

Decision:

Run a metadata/access audit before downloading or installing anything. Confirm
the gated model revisions are accessible, calculate total checkpoint size, and
retain a credible single-A6000 offload path. If this passes, run one bounded
17-frame smoke before attempting the default 93-frame horizon.

Initial checkpoint audit:

- Host requirements pass: driver `580.159.03`, glibc `2.35`, two available
  RTX A6000 48GB GPUs, and approximately `317 GB` free workspace storage.
- Cosmos-Reason1-7B metadata is accessible and totals approximately `15.45 GB`.
- Cosmos-Guardrail1 metadata is accessible and totals approximately `6.68 GB`;
  guardrails can be disabled for the feasibility smoke.
- The first audit incorrectly treated `diffusers/base/post-trained` as a path
  prefix on `main`, producing `files=0`. It is a Hugging Face revision.

Decision:

Correct the metadata audit to query the Diffusers revision directly. Do not
download or install until its total size and component layout are known.

Corrected checkpoint audit:

- `nvidia/Cosmos-Predict2.5-2B` revision `diffusers/base/post-trained` is
  accessible and totals `19.77 GiB`.
- The revision is self-contained: `15.45 GiB` text encoder, `3.84 GiB`
  transformer, `0.47 GiB` VAE, and tokenizer/scheduler metadata.
- The separate Reason1 and Guardrail repositories are not required for the
  bounded Diffusers smoke.
- Workspace storage and host requirements remain sufficient.

Decision:

The deployment audit passes. Run one isolated post-trained 2B Video2World
smoke on a single RTX A6000 using one difficult BDD left-turn clip, `17`
output frames, and `8` inference steps. This is a technical and visual
feasibility test, not a quality gate. Permit one two-A6000 `device_map`
correction only if the single-GPU attempt fails specifically due to CUDA OOM.

## 2026-06-15 - Closed Experiment Storage Cleanup

Workspace storage cleanup is restricted to directions that have already been
formally closed. Preserve source code, logs, manifests, validation frames,
current transition16 artifacts, deployed Wan/DiffSynth/LTX environments, and
all active Cosmos files.

Delete the isolated environments, checkpoints, third-party repositories, and
generated outputs for Epona, MemFlow-P, VDPM, LTX LoRA training, learned
future-flow experiments, and closed LTX history/prompt/future-anchor tuning.
The cleanup script validates every path against the experiment and third-party
roots before deletion.

## 2026-06-16 - Cosmos Predict 2.5 Access Failure

The Diffusers smoke failed before model loading:

- Hugging Face metadata remained readable, but downloading
  `diffusers/base/post-trained/model_index.json` returned `403 Forbidden`.
- The account is not on the gated model's authorized list.
- No GPU feasibility or quality inference was performed.

Decision:

Close Cosmos-Predict2.5 under the predefined access-failure stop rule. Do not
wait for access or retry authentication during the remaining schedule. Delete
the isolated Cosmos environment, partial cache, prepared smoke inputs/output,
and cloned third-party repository during closed-experiment cleanup.

## 2026-06-16 - LTX Low-Frequency Region Correction Proposal

The direct CNN spatial blend improves BDD pixel metrics but introduces
ghosting/melting and reduced the public leaderboard score. LTX transition16
retains stronger generative/FID/FVD behavior, while region prediction has
better BDD layout and pixel fidelity.

Test one bounded CPU-only frequency-separated composite using existing
outputs: keep LTX as the full-resolution base and add only the blurred
low-frequency difference between region and LTX. Evaluate three conservative
variants in one BDD24 gate. Continue only if the best variant improves LTX16
by at least `+0.15` PSNR, loses no more than `0.002` SSIM, and improves
AbsDiff. If it passes, materialize only that variant for visual QA. If it
fails, close frequency-separated compositing without further tuning.

Low-frequency BDD24 gate result:

- Best variant `sigma=16`, `alpha=0.50` improves LTX transition16 by
  `+0.4783` PSNR and `+0.0088` SSIM.
- AbsDiff improves by `-1.8475`.
- The candidate passes the pixel gate strongly.

Decision:

Do not promote from pixel metrics alone because the primary leaderboard gap is
FID/FVD. Materialize only the winning variant on validation, re-evaluate after
video encoding, create contact sheets, and compare relative image/video
distribution proxies. Promote only if it improves both proxies and visual QA
shows no new blur, ghosting, flicker, or motion suppression.

Materialized and distribution-proxy result:

- After materialization, low-frequency correction retains `+0.3467` PSNR,
  `+0.0069` SSIM, and improves AbsDiff by `-1.1632` versus transition16.
- The relative proxy is directionally validated because transition16 ranks
  better than raw LTX, matching the known leaderboard improvement.
- Low-frequency correction worsens FID proxy from `4.6059` to `5.1859`.
- It worsens FVD proxy from `21.9625` to `23.5157`.

Decision:

Close low-frequency correction. It improves pixel fidelity but repeats the
known failure mode of sacrificing distributional/video quality. Do not
materialize it on test or tune sigma/alpha further.

The leaderboard and proxy both show that early-frame transition is the only
existing change that improves pixel and distributional metrics together.
Permit one final bounded transition-duration gate at `24` and `32` frames.
Promote only if one duration improves both proxies versus transition16 without
losing more than `0.15` PSNR or `0.003` SSIM. No further duration or curve
sweep if both fail.

Transition-duration result:

- Transition24 improves PSNR by `+0.3116`, SSIM by `+0.0051`, and FID proxy
  from `4.6059` to `4.5519`, but worsens FVD proxy from `21.9625` to
  `22.4621`.
- Transition32 improves PSNR by `+0.5663` and SSIM by `+0.0093`, but worsens
  FID proxy to `4.6407` and FVD proxy to `23.4023`.

Decision:

Neither candidate improves both distribution proxies. Retain transition16 and
close transition-duration/curve tuning. The monotonic pixel improvement with
longer transitions is misleading because increasing last-frame persistence
damages video-distribution quality.

The next FID/FVD audit should target the WTS branch. WTS represents `27/71`
test samples and currently uses optical-flow warping, which can introduce
blurred edges, geometry deformation, motion saturation, and freeze. Compare
WTS candidate distributions against real validation videos before changing
the promoted BDD transition16 branch.

## 2026-06-16 - WTS Distribution Proxy Gate Proposal

Audit WTS directly on ten fixed/overhead and ten vehicle/IP validation
samples, using 50 consecutive real future frames per sample. Materialize and
compare repeat-last, region-medium, region-long12, and region-long24. This
tests whether the longer-motion WTS candidates improve temporal distribution
enough to reduce FVD or merely add warp artifacts.

Promote a WTS replacement only if it improves both FID and FVD proxies versus
region-medium, already has acceptable pixel-metric evidence from the existing
long-horizon gate, and passes visual QA. If none pass, retain region-medium
and close WTS flow-duration tuning.

WTS aggregate proxy result:

- `region_medium`: FID `5.3064`, FVD `25.7751`;
- `region_long12`: FID `5.4758`, FVD `24.2390`;
- `region_long24`: FID `5.7592`, FVD `24.3612`;
- `repeat_last`: FID `3.7217`, FVD `31.4131`.

No aggregate candidate improves both proxies. Long12 is still a bounded
routing candidate because it substantially improves FVD, only mildly worsens
FID, and previously improved long-horizon pixel metrics in both WTS domains.
Run one domain-separated proxy audit using the already materialized videos.
Promote per-domain routing only when a candidate improves both proxies within
that domain; otherwise retain region-medium and stop flow-horizon tuning.

Domain-separated result:

- overhead/fixed `region_long24` improves FID proxy from `2.5066` to `2.4931`
  and FVD proxy from `17.4377` to `17.1697`;
- vehicle/IP long12 and long24 improve FVD but materially worsen FID, so retain
  `region_medium`.

The combined route, overhead long24 plus vehicle/IP medium, improves aggregate
FID proxy from `5.3064` to `5.2959` and FVD proxy from `25.7751` to `25.6710`.
This is a small but consistent pass, and overhead long24 also improved prior
long-horizon pixel metrics. Require visual QA for warp stretching, duplicated
actors, and edge ghosting before building or packaging a test candidate.

Visual QA contact sheets showed no obvious new stretching, duplicated actors,
or edge ghosting versus region-medium. The test candidate verified all `71/71`
samples: `44` BDD transition16 and `27` WTS routed outputs. Promote this as a
bounded leaderboard candidate; retain the previous `68.2611` ZIP as rollback
because the expected gain is small.

## 2026-06-16 - Perceptual And Video Distribution Audit Proposal

The latest leaderboard comparison shows that the primary gap to rank 1 is
distributional/perceptual quality rather than PSNR or SSIM:

- current: `FID 50.2430`, `FVD 32.1304`, `LPIPS 0.3393`;
- rank 1: `FID 31.1354`, `FVD 22.4018`, `LPIPS 0.2727`;
- rank 1 has slightly lower SSIM than the current submission.

Therefore do not promote further pixel-space corrections using PSNR/SSIM
alone. Add a local relative proxy audit using Inception image features and
3D-ResNet video features on the same BDD24 validation set. The proxy is useful
only if it correctly ranks transition16 above raw LTX, matching the known
leaderboard improvement. Future candidates must improve both distribution
proxies and avoid material pixel-metric regressions before submission.


## 2026-06-16 - Leaderboard Result: WTS Domain-Routed Candidate

Submission `hybrid_ltx_transition16_wts_domain_routed_submission` scored:

- Final score `68.3249` vs previous `68.2611` (`+0.0638`);
- PSNR `18.2307` vs `18.2169` (`+0.0138`);
- SSIM `0.6113` vs `0.6114` (`-0.0001`);
- LPIPS `0.3394` vs `0.3393` (`+0.0001`, slightly worse);
- CLIP `0.9351` vs `0.9348` (`+0.0003`);
- FID `49.9872` vs `50.2430` (`-0.2558`);
- FVD `31.7069` vs `32.1304` (`-0.4235`).

Decision: keep this as the current best submission, but close further WTS
flow-horizon tuning. The leaderboard gain matches the proxy direction but is
too small to justify more local horizon/decay sweeps. The remaining gap is
mainly perceptual/distributional quality, especially LPIPS/FID/FVD, and needs a
larger mechanism than WTS flow routing.

Next bounded check: BDD transition-duration routing by phase. Aggregate
transition24 improved FID but hurt FVD, and transition32 hurt both distribution
proxies despite better pixel metrics. Because phase is available in the test
metadata, run one phase-separated proxy audit using existing transition16/24/32
validation videos. Only continue if a phase-level route improves both FID and
FVD proxies, followed by an aggregate routed proxy pass. Do not route by
individual validation sample ID.


## 2026-06-16 - WTS Actor/Vehicle Anti-Smear Gate Proposal

Visual QA confirms that the main WTS artifact is not global background quality
but actor/vehicle smear from repeated optical-flow warping: pedestrians and
cars become blurry, stretched, or double-imaged while the static scene remains
mostly stable.

Test one bounded hand-engineered correction: keep region-flow for the scene but
blend suspected foreground moving regions back toward the last observed frame.
Evaluate `region_actor_guard25`, `region_actor_guard50`, and
`region_long24_actor_guard50` against repeat-last, region-medium, and
region-long24 on the same 20 WTS validation clips used by the distribution
proxy gate.

Continue only if an actor-guard variant improves both FID and FVD proxies and
visual QA does not show frozen pedestrians/vehicles or mask halos. If it fails,
close hand-engineered WTS anti-smear compositing rather than tuning masks.


Actor-guard aggregate proxy result:

- `region_medium`: FID `5.3064`, FVD `25.7751`;
- `region_long24`: FID `5.7592`, FVD `24.3612`;
- `guard25`: FID `4.8921`, FVD `25.6739`;
- `guard50`: FID `4.6173`, FVD `25.9540`;
- `long24_guard50`: FID `4.9087`, FVD `25.0384`;
- `repeat_last`: FID `3.7217`, FVD `31.4131`.

`long24_guard50` improves both aggregate proxies versus the current routed WTS
baseline (`5.2959` FID, `25.6710` FVD), and it directly targets the visible
WTS smear artifact. Continue to domain-separated proxy audit and visual QA.


Actor-guard routed proxy result:

- current overhead-long24/vehicle-medium: FID `5.2959`, FVD `25.6710`;
- vehicle `guard25`: FID `4.8836`, FVD `25.5352`;
- vehicle `guard50`: FID `4.6111`, FVD `25.8090`;
- vehicle `long24_guard50`: FID `4.9126`, FVD `24.9061`.

`vehicle_long24_guard50` is the best aggregate proxy route, but visual QA must
check halos carefully. Initial inspection shows long24-style green/edge smear
can appear in vehicle/IP scenes; if `guard25` is visually cleaner, prefer it as
the safer candidate despite the smaller FVD gain. Do not build test until
vehicle/IP guard visual QA passes.

Decision: promote the safer `guard25` vehicle/IP route for one test candidate.
It improves both routed distribution proxies versus the current WTS route
without using the visually riskier long24 horizon on vehicle/IP:

- current overhead-long24/vehicle-medium: FID `5.2959`, FVD `25.6710`;
- vehicle `guard25`: FID `4.8836`, FVD `25.5352`.

Implementation adds `--region_actor_guard_strength` to the baseline generator.
Default is `0.0`, so existing region-flow submissions are unchanged. Build one
candidate with BDD `ltx_transition16`, WTS overhead/fixed `region_long24`, and
WTS vehicle/IP `region_actor_guard25`; do not zip on server.


## 2026-06-16 - BDD Transition Phase Proxy Gate

Phase-level transition-duration proxy found one possible route and four
rejections:

- phase 0: `transition24` and `transition32` both improve FID/FVD proxies versus
  `transition16`; `transition32` has better FID, `transition24` has slightly
  better FVD.
- phases 1, 2, 3, 4: `transition16` remains best on both FID/FVD proxies.

Continue only with an aggregate routed proxy gate: phase 0 uses transition24 or
transition32, all other BDD phases use transition16. Do not route by validation
sample ID.

Aggregate phase-route proxy result:

- `transition16`: FID `4.6059`, FVD `21.9625`;
- `phase0_route24`: FID `4.5702`, FVD `21.5339`;
- `phase0_route32`: FID `4.5365`, FVD `21.4572`.

Promote `phase0_route32` to one test candidate because it improves both
aggregate FID and FVD proxies. Keep the WTS side unchanged at the current best
domain route: overhead/fixed `region_long24`, vehicle/IP `region_medium`.

Leaderboard result: no score change from the current best. Audit shows the
public test index has no BDD phase 0 rows (`bdd` counts: phase1=12, phase2=3,
phase3=15, phase4=14), so the phase0 route wrote zero transition32 rows and the
candidate is byte-equivalent in routing to the current best for public test.
Close BDD phase0 routing for this public submission set.


## 2026-06-16 - Minimal Prompt + Low-Frequency Combination Smoke

Tested a very short BDD LTX prompt and the earlier low-frequency correction on
top of that prompt using four prompt-sensitive validation samples.

Pixel metrics:

- `legacy_blank`: PSNR `17.5370`, SSIM `0.7199`;
- `minimal`: PSNR `17.5225`, SSIM `0.7209`;
- `transition16`: PSNR `18.2498`, SSIM `0.7230`;
- `minimal_lowfreq`: PSNR `18.5422`, SSIM `0.7424`.

Distribution proxies:

- `transition16`: FID `9.7042`, FVD `48.0777`;
- `minimal`: FID `10.2004`, FVD `51.4560`;
- `minimal_lowfreq`: FID `12.1028`, FVD `50.5956`;
- `legacy_blank`: FID `10.2746`, FVD `59.0171`.

Decision: reject both `minimal` and `minimal_lowfreq`. Low-frequency correction
again improves PSNR/SSIM but worsens FID materially. Retain current
`transition16` prompt behavior and close prompt/low-frequency combination work.


## 2026-06-17 - WTS Object-Layer Motion Gate

Tested a non-generative object-layer alternative for WTS small-object motion:
median-history background plus foreground moving-layer transport. This uses
only input-frame statistics, not public sample IDs.

Initial routed proxy:

- current overhead-long24/vehicle-medium: FID `5.2959`, FVD `25.6710`;
- vehicle `actor_layer_filled`: FID `5.0845`, FVD `22.2625`;
- overhead `actor_layer_filled`: FID `5.2836`, FVD `24.4240`.

Visual QA rejected vehicle `actor_layer_filled`: it can treat dashboard
reflections and large foreground regions as actors, creating obvious
copy-paste/halo artifacts.

Small-blob correction:

- vehicle `small_filled`: FID `3.6496`, FVD `32.9785`, rejected because FVD is
  much worse;
- overhead `small_filled`: FID `5.2838`, FVD `24.4441`, passes proxy lightly and
  visual QA on overhead sheets did not show the large vehicle/reflection
  artifacts.

Build one bounded test candidate with BDD `transition16`, WTS overhead/fixed
`actor_layer_small_filled`, and WTS vehicle/IP current `region_medium`.

Full WTS robustness gate hit an incomplete validation row:
`val_wts_main_wts_vehicle_or_ip_camera_20230922_44_CN4_T2_20230922_44_CN4_T2_vehicle_view_phase4`
had `targets=0` for a 50-frame full-video materialization. This is a manifest /
frame-availability issue, not a model failure. Updated
`materialize_wts_distribution_gate.py` so it selects rows only after confirming
conditions and requested target frames exist, skips incomplete rows with a CSV
audit, and continues until it has the requested valid rows per WTS domain.

The 50/domain full gate is not feasible with 50 future frames in the current
balanced manifest: after skipping incomplete vehicle/IP rows, only 45 valid
vehicle/IP rows remain. Updated `run_wts_object_layer_full_proxy_gate.sh` to use
45/domain by default (`full90`). This is still a larger robustness gate than
the 20-clip smoke and avoids silently shortening/evaluating missing clips.

Full90 result:

- proxy: `overhead_small_filled` FVD improves `16.3848 -> 15.8448`, while FID
  slightly worsens `3.0361 -> 3.0510`;
- pixel overall: PSNR drops `25.6395 -> 25.5422`, SSIM drops
  `0.9261 -> 0.9235`, AbsDiff worsens `9.0955 -> 9.1059`;
- overhead-only pixel: PSNR drops `29.3839 -> 29.1893`, SSIM drops
  `0.9593 -> 0.9541`;
- vehicle/IP is unchanged by construction.

Decision: full90 partially supports the submitted overhead-object candidate for
distribution/FVD, but the pixel loss is real. Treat it as a bounded leaderboard
probe, not a replacement for the current best unless public score improves.

Because time is short, continue without waiting for leaderboard. Added a
general guarded route for the overhead object-layer direction:

- `materialize_wts_distribution_gate.py` now writes actor mask statistics for
  `actor_layer_small_filled`;
- `build_wts_actor_guarded_report.py` routes overhead samples to
  `actor_layer_small_filled` only when actor mask coverage/component stats are
  below thresholds, otherwise falls back to current `region_long24`;
- `run_wts_object_layer_guarded_full90_gate.sh` tests `guard005`, `guard010`,
  and `guard015` on full90.

Promotion requires retaining FVD improvement while reducing the pixel loss seen
in the unguarded overhead-small candidate.


## 2026-06-18 - BDD LTX Early Transition Ghosting

Visual QA found a consistent BDD LTX artifact: the first few generated frames
after the conditioning window show double exposure / blur, then stabilize. This
matches the current `transition16` implementation, which alpha-blends the last
conditioning frame into raw LTX frames. When the camera or actors move, that
blend creates transparent ghosts.

Added a bounded general gate, not tied to public sample IDs:

- `materialize_ltx_flow_bridge.py` builds early `region_predictions` from the
  last two conditioning frames and hands off to raw LTX.
- `run_ltx_flow_bridge_gate.sh` tests `flow_bridge4_blend4`,
  `flow_bridge8_blend4`, and `flow_bridge4_hard` against current
  `transition16`.

Promotion requires visible reduction of first-frame ghosting, no material
pixel-metric loss, and comparable or better FVD proxy. If it fails, retain
`transition16` and close this bridge direction.

Gate result:

- `transition16`: PSNR `17.6807`, SSIM `0.7699`, FID proxy `4.6059`, FVD proxy
  `21.9625`;
- `flow_bridge4_blend4`: PSNR `17.3700`, SSIM `0.7629`, FID proxy `4.7426`,
  FVD proxy `21.4821`;
- `flow_bridge8_blend4`: PSNR `17.6315`, SSIM `0.7666`, FID proxy `4.7325`,
  FVD proxy `21.9517`;
- `flow_bridge4_hard`: PSNR `17.1249`, SSIM `0.7574`, FID proxy `4.8373`,
  FVD proxy `22.3076`.

Decision: reject the flow bridge. `bridge4_blend4` improves FVD but loses too
much PSNR/SSIM and worsens FID; `bridge8_blend4` keeps pixels closer but barely
improves FVD. Visual sheets still show early blur/warp artifacts, so this does
not solve the transition ghosting cleanly.

Next bounded correction: motion/edge masked transition. Instead of replacing
early frames with optical-flow predictions, keep the existing `transition16`
structure but increase alpha toward raw LTX earlier in high-motion / high-edge
regions. This targets double exposure while avoiding flow-warp artifacts.

Added:

- `materialize_ltx_masked_transition.py`
- `run_ltx_masked_transition_gate.sh`

The gate tests `mask_a25`, `mask_a40`, and `mask_a60` against `transition16` on
the 24-sample BDD dense validation set with pixel metrics, FID/FVD proxy, and
contact sheets.

Gate result:

- `transition16`: PSNR `17.6807`, SSIM `0.7699`, FID proxy `4.6059`, FVD proxy
  `21.9625`;
- `mask_a25`: PSNR `17.6000`, SSIM `0.7695`, FID proxy `4.8060`, FVD proxy
  `22.5773`;
- `mask_a40`: PSNR `17.5388`, SSIM `0.7688`, FID proxy `4.9166`, FVD proxy
  `23.1939`;
- `mask_a60`: PSNR `17.4429`, SSIM `0.7671`, FID proxy `4.9814`, FVD proxy
  `23.7909`.

Decision: reject and close masked transition. All mask strengths worsen both
FID and FVD proxies, and pixel metrics degrade monotonically. Retain
`transition16`.

## 2026-06-20 - Temporal Latent Train-On-Train Gate

Goal:

Run the temporal latent feature refiner on train data before any train300
scale-up.

Data repair:

The prepared script could not run as-is because
`train_ticft_balanced_300_frames.jsonl` points to the missing
`/workspace/experiments/wan_ticft/frames_balanced_300` directory. Added
`wan_ticft/data/materialize_manifest_frames.py` and rematerialized BDD
condition/target PNGs from the source train videos. The balanced-300 manifest
contains only 60 BDD rows, so this is effectively a smaller train gate.

Result:

```text
cnn_ltxmix65        PSNR 18.2129  SSIM 0.7877  AbsDiff 22.1611  FID 4.9677  FVD 22.2935
temporal_train100   PSNR 18.0991  SSIM 0.7852  AbsDiff 22.3894  FID 5.2649  FVD 22.5984
```

Visual findings:

Contact/comparison sheets show the refiner sometimes reduces blur/ghosting, but
more often applies small residuals that darken, soften, or slightly misalign
frames that `cnn_ltxmix65` already predicts better.

Decision:

Gate failed. Do not scale this temporal latent train-on-train run to train300.
The main issue is likely the mismatch between training on region-motion base
and evaluating on `cnn_ltxmix65`. If revisited, first create a BDD train base
that matches `cnn_ltxmix65` behavior, then rerun the gate.


## 2026-06-18 - CNN Selector Perceptual Tradeoff

Leaderboard probe `cnn` scored high pixel metrics but poor distribution:
PSNR `19.2046`, SSIM `0.6229`, LPIPS `0.3159`, but FID `58.4865` and FVD
`35.2033`. This matches the architecture: the CNN selector leans heavily toward
region/flow predictions, which improve pixel alignment but look less natural
than LTX.

Added `materialize_cnn_ltx_region_selector_variants.py` and
`run_cnn_selector_perceptual_tradeoff_gate.sh` to test whether mixing CNN output
back toward LTX (`ltx_mix=0.25/0.50/0.65`) can preserve useful PSNR gain while
reducing FID/FVD proxy.

Validation tradeoff result:

- `cnn_ltxmix0`: PSNR `18.9101`, SSIM `0.8008`, FID proxy `7.9479`, FVD proxy
  `29.2870`;
- `cnn_ltxmix25`: PSNR `18.7534`, SSIM `0.8002`, FID proxy `6.7718`, FVD proxy
  `25.2470`;
- `cnn_ltxmix50`: PSNR `18.4737`, SSIM `0.7944`, FID proxy `5.5280`, FVD proxy
  `22.6146`;
- `cnn_ltxmix65`: PSNR `18.2152`, SSIM `0.7879`, FID proxy `4.9636`, FVD proxy
  `22.2482`;
- `transition16`: PSNR `17.6807`, SSIM `0.7699`, FID proxy `4.6059`, FVD proxy
  `21.9625`.

Decision: build one leaderboard probe with `cnn_ltxmix65`. It keeps a meaningful
pixel gain over `transition16` while largely reducing raw CNN's proxy damage.
It is still distribution-worse than pure LTX, so this is a score-tradeoff
submission, not a clean promotion.

Added `materialize_cnn_ltxmix_test.py` and
`build_cnn_ltxmix65_wts_overhead_object_layer_candidate.sh`.

Follow-up sweep around the CNN/LTX mix point:

- `mix45`: PSNR `18.4152`, SSIM `0.7939`, FID proxy `5.8233`, FVD proxy
  `23.6330`;
- `mix55`: PSNR `18.2906`, SSIM `0.7907`, FID proxy `5.3565`, FVD proxy
  `22.6069`;
- `mix60`: PSNR `18.2278`, SSIM `0.7890`, FID proxy `5.1606`, FVD proxy
  `22.2565`;
- `mix65`: PSNR `18.2129`, SSIM `0.7877`, FID proxy `4.9677`, FVD proxy
  `22.2935`;
- `mix70`: PSNR `18.0705`, SSIM `0.7845`, FID proxy `4.9257`, FVD proxy
  `22.4122`;
- `mix75`: PSNR `17.9947`, SSIM `0.7822`, FID proxy `4.8547`, FVD proxy
  `22.3844`.

Decision: `mix45` is the only aggressive variant worth a leaderboard probe. It
has the largest pixel gain while remaining far less distribution-damaging than
raw CNN. `mix60/65/70/75` are too close to justify multiple submissions unless
`mix45` fails badly.

Leaderboard result for `cnn_ltxmix65` with overhead-object WTS:

- Score `69.1436`;
- PSNR `18.5786`;
- SSIM `0.6222`;
- LPIPS `0.3295`;
- CLIP `0.9390`;
- FID `48.5029`;
- FVD `31.2674`.

This is a substantial improvement over the previous best `68.3523`. Continue
with `cnn_ltxmix65` as the BDD base and submit the stricter WTS `guard015`
variant next, because full90 validation showed `guard015` improves FVD and
pixel metrics versus current/unguarded WTS.

## 2026-06-24 - LTX 2B FrameInit Smoke

Tested an opt-in FrameInit-style latent initialization in LTX 2B. The patch
injects low-frequency spatial structure from the last observed conditioning
latent into future latent noise while leaving high-frequency content random.
It is disabled unless `LTX_FRAMEINIT=1`.

BDD2 smoke result:

```text
depth_a75     PSNR 16.4793  SSIM 0.7663  AbsDiff 24.7860  FID  8.3740  FVD 22.7012
ltx2b_base    PSNR 16.3091  SSIM 0.7598  AbsDiff 25.4941  FID  8.8063  FVD 25.5961
frameinit     PSNR 14.2346  SSIM 0.6632  AbsDiff 38.0648  FID 11.9568  FVD 39.8903
repeat_last   PSNR 15.6776  SSIM 0.6813  AbsDiff 28.0577
```

Outputs:

```text
/workspace/experiments/wan_ticft/ltx2b_frameinit_s035_k9_bdd2_eval.csv
/workspace/experiments/wan_ticft/ltx2b_frameinit_s035_k9_bdd2_proxy.csv
/workspace/experiments/wan_ticft/ltx2b_frameinit_s035_k9_bdd2_compare_sheets
```

Decision:

Reject `s035_k9` immediately. It worsens pixel metrics and heavily worsens
FID/FVD proxy, so do not run BDD24 or any submission build from this variant.
Keep the patch only as opt-in repro code; default LTX behavior remains
unchanged.

## 2026-06-25 - Vista IMG Last-Frame BDD4 Gate

Vista official `IMG` mode was tested as a last-frame-anchor BDD world-model
candidate. The wrapper writes normal project `report.csv` outputs and compares
against `ltx2b_base` and `depth_a75`.

BDD4 result:

```text
depth_a75     PSNR 16.2962  SSIM 0.7610  AbsDiff 26.3280  FID  8.6556  FVD 31.7490
ltx2b_base    PSNR 16.1623  SSIM 0.7556  AbsDiff 26.9257  FID  8.9253  FVD 33.3218
vista_img     PSNR 15.8087  SSIM 0.7409  AbsDiff 27.6220  FID 11.3147  FVD 33.2921
repeat_last   PSNR 15.6917  SSIM 0.7058  AbsDiff 28.1382
```

Decision:

Reject raw Vista `IMG`: it worsens FID and loses pixel fidelity. Do not scale
to BDD24 or submission. Continue with temporal VSR/detailing on the `depth_a75`
scaffold.

## 2026-06-25 - StableVSR Detail On Depth-A75 BDD4

StableVSR was tested as a temporal diffusion detailer on `depth_a75`: downscale
the depth scaffold to `320x180`, restore with StableVSR x4, then score the
materialized `1280x720` report.

BDD4 result:

```text
depth_a75     PSNR 16.2962  SSIM 0.7610  AbsDiff 26.3280  FID 8.6556  FVD 31.7490
stablevsr     PSNR 16.2464  SSIM 0.7610  AbsDiff 26.7402  FID 8.8793  FVD 30.1516
ltx2b_base    PSNR 16.1623  SSIM 0.7556  AbsDiff 26.9257  FID 8.9253  FVD 33.3218
repeat_last   PSNR 15.6917  SSIM 0.7058  AbsDiff 28.1382
```

Visual QA: geometry and actor identity are mostly preserved, but the output has
a slightly processed contrast/texture. Decision: borderline continue, not a
promotion. Try a less-aggressive `480x270`, `STEPS=10` variant before any BDD24
scale-up.

Follow-up `480x270`, `STEPS=10` result:

```text
depth_a75     PSNR 16.2962  SSIM 0.7610  AbsDiff 26.3280  FID 8.6556  FVD 31.7490
stablevsr     PSNR 16.3379  SSIM 0.7650  AbsDiff 26.5081  FID 8.8440  FVD 32.7494
ltx2b_base    PSNR 16.1623  SSIM 0.7556  AbsDiff 26.9257  FID 8.9253  FVD 33.3218
repeat_last   PSNR 15.6917  SSIM 0.7058  AbsDiff 28.1382
```

Decision update: do not scale StableVSR. The first variant improved FVD but
worsened FID; the second improved pixel metrics but worsened both distribution
proxies versus `depth_a75`. Move to a Track5-trained scaffold-conditioned
residual/detail model.

## 2026-06-25 - Prepared LTX2B Depth-Scaffold LoRA Smoke

Prepared the next route: train a small LTX2B IC-LoRA to refine `depth_a75`
scaffold videos into ground-truth future videos. This is not full generator
pretraining; it is scaffold-conditioned domain adaptation.

Files added:

- `wan_ticft/evaluation/materialize_manifest_target_report.py`
- `wan_ticft/evaluation/score_ltx_scaffold_lora_validation.py`
- `wan_ticft/evaluation/run_ltx2b_depth_scaffold_lora_bdd_smoke.sh`

The smoke runner builds/resumes BDD train LTX base, materializes depth-a75
train/eval scaffold, writes GT target reports, prepares paired IC-LoRA data,
trains 200 steps by default, then scores validation samples against scaffold.

Run:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_ltx2b_depth_scaffold_lora_bdd_smoke.sh
```

Promotion gate: only continue if LoRA checkpoint samples improve FID/FVD proxy
or LPIPS-like visual quality over `depth_a75` without geometry hallucination or
large PSNR/SSIM loss.

Runtime fix: first run failed during LTX-Video-Trainer preprocessing because
the trainer now requires dataset media paths to be relative to `dataset.json`'s
parent. `prepare_ltx_ic_lora_dataset.py` now stages target/reference videos
under `ic_lora_data/media/...` and writes relative `media_path` /
`reference_path`; the runner's text-embedding repair was made conditional.

BDD4 validation result:

```text
scaffold  PSNR 16.8121  SSIM 0.7814  AbsDiff 24.0446  FID  9.4618  FVD 17.9846
step0     PSNR 11.3499  SSIM 0.2764  AbsDiff 49.6081  FID 19.4067  FVD 45.4499
step100   PSNR 10.7612  SSIM 0.2190  AbsDiff 54.4214  FID 24.4240  FVD 50.3172
step200   PSNR 10.7035  SSIM 0.2108  AbsDiff 54.8376  FID 24.7332  FVD 46.7189
```

Decision: reject current LTX IC-LoRA scaffold route. Since `step0` already
collapses relative to the scaffold, the official reference-video path is not a
copy-preserving refiner. Do not run BDD24 or submission from this route.

## 2026-06-25 - BDD Phase12 Repeat/Depth Route

Oracle/headroom audit showed useful routing signal between `depth_a75` and the
repeat-like `convex_repeat_depth` candidate:

```text
depth_a75  PSNR 18.5623  SSIM 0.8344  AbsDiff 19.0541  FID 5.8550  FVD 19.2549
oracle     PSNR 20.6166  SSIM 0.8516  AbsDiff 15.4971  FID 5.0707  FVD 19.4259
```

All-in convex/repeat worsens FVD badly, so use narrow routing. The simple rule
`phase1+phase2 -> repeat/convex, otherwise depth_a75` passes BDD24 proxy:

```text
phase12    PSNR 19.9558  SSIM 0.8397  AbsDiff 17.3814  FID 4.9540  FVD 17.3987
depth_a75  PSNR 18.5623  SSIM 0.8344  AbsDiff 19.0541  FID 5.8550  FVD 19.2549
```

Prepared test candidate builder:

- `wan_ticft/evaluation/materialize_repeat_last_test.py`
- `wan_ticft/evaluation/materialize_phase_rule_test.py`
- `wan_ticft/evaluation/build_bdd_phase12_repeat_depth_wts_overhead_candidate.sh`

This builds BDD phase1/2 repeat-depth routing and keeps current best WTS
overhead/object-layer outputs unchanged.
