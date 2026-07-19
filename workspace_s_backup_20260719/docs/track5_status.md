# AI City Track 5 Handoff

Last updated: 2026-06-06

## Goal

Generate future traffic-video frames from:

- history frames;
- `caption_pedestrian`;
- `caption_vehicle`;
- requested future frame count `target_N`.

The final submission requires exactly `target_N` frames per test sample at
`1280x720`.

## Current Best Submitted Baseline

The submitted baseline is official Wan2.2 TI2V-5B zero-shot inference.

Leaderboard result on the published 50% test subset:

| Metric | Result |
| --- | ---: |
| Final Score | 64.3031 |
| PSNR | 16.4542 |
| SSIM | 0.4820 |
| LPIPS | 0.3593 |
| CLIP | 0.9219 |
| FID | 52.0823 |
| FVD | 38.4845 |

Interpretation:

- CLIP and FVD are strong: generated videos look realistic and match captions.
- PSNR and SSIM are weak: generated futures often do not match the true scene
  and motion closely enough.
- The next system should preserve Wan semantic/video quality while improving
  scene and motion fidelity.

The local flat submission archive was created as:

```text
G:\prediction_flat.zip
```

Its archive root directly contains 71 sample folders. It has 5194 PNG files,
uses frame names `0.png`, `1.png`, etc., and has no extra `prediction/` folder.

## Server Environment

Workspace:

```text
/workspace
```

Important repositories and packages:

```text
/workspace/baseline
/workspace/wan_ticft
/workspace/third_party/Wan2.2
/workspace/third_party/DiffSynth-Studio
/workspace/models/Wan2.2-TI2V-5B
```

Conda environments:

- `wan22`: Wan2.2/DiffSynth.
- `codex-cli`: Codex CLI and Node.js.

Wan environment confirmed:

- PyTorch `2.5.1+cu121`
- Diffusers `0.38.0`
- Transformers `4.51.3`
- Accelerate `1.13.0`
- Flash Attention `2.8.3`
- 2 A100 40GB GPUs at IDs 0 and 1

## Existing Code

### Baseline package

Useful modules:

```text
baseline/submission.py
baseline/video_contact_sheet.py
baseline/diagnose_motion.py
baseline/train_refiner.py
baseline/eval_refiner.py
baseline/infer_refiner.py
```

The residual-flow refiner improves validation PSNR/SSIM but long rollout
generation tends to freeze or drift. Keep it as a diagnostic or possible
fixed-camera/pixel-preserving candidate, not the only generator.

### Wan TIC-FT package

Data:

```text
wan_ticft/data/build_manifest.py
wan_ticft/data/verify_manifest.py
wan_ticft/data/frame_sampling.py
```

Training:

```text
wan_ticft/training/prepare_diffsynth_dataset.py
wan_ticft/training/layout.py
wan_ticft/training/train_ticft.py
```

Inference:

```text
wan_ticft/inference/run_wan22_ti2v.py
wan_ticft/inference/run_wan22_ti2v_parallel.py
wan_ticft/inference/run_diffsynth_lora_ti2v.py
wan_ticft/inference/videos_to_submission_frames.py
```

Important: `run_diffsynth_lora_ti2v.py` must import:

```python
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
```

The installed DiffSynth version does not contain `wan_video_new`.

## Completed Experiments

### Official Wan2.2 zero-shot

Full test generation:

```text
/workspace/experiments/wan_ticft/wan22_ti2v_test_121f_s10
```

Results:

- 71/71 generated successfully.
- 121 frames, 10 sampling steps.
- Official Wan repository `generate.py --task ti2v-5B` was used.
- Converted into 5194 submission frames successfully.
- This produced the leaderboard baseline score listed above.

### Residual-flow refiner

Checkpoint:

```text
/workspace/experiments/refiner_mixed_h1_16_v1/refiner_best.pt
```

Validation improvements included approximately:

- horizon 8: `+1.98 dPSNR`, `+0.071 dSSIM`;
- horizon 4: `+1.96 dPSNR`, `+0.062 dSSIM`.

However, full predictions were diagnosed as effectively frozen. Rollout and
anchor-rollout variants did not resolve this sufficiently.

### DiffSynth LoRA 13 frames

This was a proof that DiffSynth training and LoRA inference work. It is too
short and conservative for the final task.

### DiffSynth LoRA 49 frames

Checkpoint directory:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_300_49f
```

Do not use this branch for final generation.

The original manifest had only 9 target frames. Preparing a 49-frame clip
created:

```text
[last condition frame] + [9 target frames] + [repeated final target frame]
```

This trained the model to freeze. Inference at 121 frames was also outside the
training horizon and produced sliding/standing actors.

### DiffSynth LoRA 121-frame smoke

Twenty 121-frame clips trained successfully without OOM:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_smoke_121f
```

This improved scene stability slightly but still produced conservative actor
motion. The smoke dataset was too small for a final conclusion.

## Latest Experiment: Balanced 500-Sample 121-Frame LoRA

Manifest:

```text
/workspace/experiments/wan_ticft/manifests/train_ticft_balanced_121f_frames.jsonl
```

Manifest verification:

- rows: 500
- bad: 0
- condition files: 2000
- target files: 60000
- zero-byte files: 0

Prepared DiffSynth dataset:

```text
/workspace/experiments/wan_ticft/diffsynth_train_balanced_500_121f
```

Prepared dataset verification:

- 500 videos
- approximately 1007 MB
- 121 frames per video
- `832x480`

Training output:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5
```

Training configuration:

- 500 samples
- 121 frames
- 1 epoch
- learning rate `5e-5`
- LoRA rank 32
- target modules `q,k,v,o,ffn.0,ffn.2`
- input image conditioning enabled through `--extra_inputs input_image`
- gradient checkpointing enabled

Training completed:

```text
500/500 in 1:35:33
```

The checkpoint files have not yet been inspected in this handoff. Start with:

```bash
find /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5 \
  -maxdepth 2 -type f \
  -printf '%p | %s bytes\n' | sort
```

## Immediate Next Steps

### 1. Validate balanced 121f LoRA on BDD

Assuming `step-500.safetensors` exists:

```bash
cd /workspace

PYTHONPATH=/workspace:/workspace/third_party/DiffSynth-Studio \
CUDA_VISIBLE_DEVICES=0 \
python -m wan_ticft.inference.run_diffsynth_lora_ti2v \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_dir /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12 \
  --lora /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5/step-500.safetensors \
  --limit 12 \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --steps 10 \
  --lora_alpha 1.0
```

Create contact sheets:

```bash
python -m baseline.video_contact_sheet \
  --input /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12/videos \
  --mode videos \
  --out_dir /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_sheets \
  --samples 12 \
  --tile_width 320 \
  --cols 4
```

### 2. Validate view-specific samples

The first balanced validation rows can be mostly BDD. Also validate:

- `wts_vehicle_or_ip_camera`;
- `wts_overhead_or_fixed_camera`.

Create filtered manifests if necessary. Do not decide on full test generation
from BDD-only validation.

### 3. Compare checkpoint strengths

Before full test generation, compare at least:

- LoRA alpha `1.0`;
- LoRA alpha `0.5` or `0.7`;
- optionally checkpoints `step-300`, `step-400`, `step-500`.

Lower alpha may recover base-Wan motion while preserving learned scene
constraints.

### 4. Build multi-seed official Wan candidate generation

The strongest practical next system is an ensemble:

```text
prompt compiler
 -> official Wan2.2 multi-seed candidates
 -> optional balanced 121f LoRA candidate
 -> optional flow/refiner candidate for fixed cameras
 -> candidate reranker
 -> submission formatter
```

Use the official Wan repository generator for the main candidates because its
TI2V image conditioning is confirmed.

## Proposed Reranking

Do not rank mainly by CLIP. The zero-shot baseline already has high CLIP while
pixel metrics are weak.

Candidate score should combine:

- first-frame consistency with the final history frame;
- background consistency for fixed overhead cameras;
- scene-change/hallucination penalty;
- freeze penalty;
- excessive-motion/drift penalty;
- actor motion plausibility;
- prompt/CLIP similarity as a lower-weight semantic term.

Use view-specific scoring:

- fixed overhead: strongly preserve background and camera;
- vehicle/IP camera: permit moderate camera/road motion;
- BDD ego view: permit strong forward motion while penalizing layout changes.

## Proposed Prompt/VLM Pipeline

Long-term pipeline:

```text
history frames + captions
 -> VLM analyzer
 -> deterministic caption compiler
 -> constrained prompt generator
 -> multi-seed Wan generation
 -> candidate reranking
 -> optional fixed-camera actor/background compositing
 -> submission formatting
```

Sa2VA is a possible VLM/segmentation model, but do not begin with a large VLM
integration. Implement the deterministic prompt compiler and reranker first.
Sa2VA/SAM2 is most useful later for actor masks, motion scoring, duplication
detection, and fixed-background compositing.

## Known Issues and Lessons

- Do not infer 121 frames from a LoRA trained mostly on padded 49-frame clips.
- Do not use only contact-sheet appearance as validation. Measure freeze,
  motion, consistency, and visual quality.
- Prompts are currently long and sometimes truncate mid-sentence. A prompt
  compiler should produce shorter, structured prompts.
- Test frame folders downloaded locally may contain zero-byte placeholders.
  Trust verified server manifests and generated reports.
- The submission portal previously rejected some ZIP uploads due to MIME/layout.
  The direct-root flat ZIP format is the current preferred format.



## 2026-06-06 Balanced 121f LoRA BDD Validation Update

Existing validation output:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12
```

Verified:

- 12/12 BDD videos succeeded; every video has 121 frames at `832x480`, 15 FPS.
- All five checkpoints from step 100 through step 500 are readable and finite.
- Motion diagnostics are stored in `motion_diagnostics.csv`.
- Mean adjacent-frame absolute difference: `5.722`.
- Mean first-to-last absolute difference: `33.151`.
- Freeze/weak-motion samples: 1/12.
- Excessive-motion/drift candidates: 2/12.

Quantitatively suspicious samples:

- `val_bdd_pc_5k_video3237_phase0`: clear near-freeze.
- `val_bdd_pc_5k_video8125_phase1`: excessive adjacent-frame change.
- `val_bdd_pc_5k_video6_phase1`: excessive motion and first-to-last drift.
- Phase-2 samples `video8139` and `video8277` require close visual review
  because their captions describe a stopped vehicle.

Dense 24-frame sheets were created at:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_val12_dense24/sheets
```

The 12 original sheets were later directly reviewed through the Windows-mounted
server workspace. Visual review is complete for this BDD alpha `1.0` run.

Visual summary:

- Stronger outputs: `video2149_phase0`, `video8125_phase1`.
- Near-freeze: `video3237_phase0`.
- Severe actor hallucination or enlargement: `video7040_phase0`,
  `video1407_phase0`, `video6846_phase1`, `video9213_phase1`,
  `video8277_phase2`.
- Severe vehicle/scene deformation or drift: `video6180_phase0`,
  `video6_phase1`.
- `video2907_phase1` is coherent but overly conservative.
- `video8139_phase2` is stable but has unnatural pedestrian persistence.

Decision:

- Do not run full test inference with step 500 at alpha `1.0`.
- Alpha `1.0` is worth continuing only as a candidate while mixed-view
  validation and lower-alpha comparisons are performed.
- Compared with documented official Wan behavior, this LoRA is not
  systematically frozen like the refiner, but alpha `1.0` does not reliably
  reduce hallucination. At least six of twelve BDD samples have severe visual
  failures, so it is unsuitable as the sole final generator.

Exact next mixed-view commands:

```bash
cd /workspace

PYTHONPATH=/workspace:/workspace/third_party/DiffSynth-Studio \
CUDA_VISIBLE_DEVICES=0 \
python -m wan_ticft.inference.run_diffsynth_lora_ti2v \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_vehicle_only.jsonl \
  --out_dir /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_vehicle4_a10 \
  --lora /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5/step-500.safetensors \
  --limit 4 --height 480 --width 832 --num_frames 121 --steps 10 --lora_alpha 1.0

PYTHONPATH=/workspace:/workspace/third_party/DiffSynth-Studio \
CUDA_VISIBLE_DEVICES=0 \
python -m wan_ticft.inference.run_diffsynth_lora_ti2v \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_dir /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_fixed4_a10 \
  --lora /workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5/step-500.safetensors \
  --offset 25 --limit 4 --height 480 --width 832 --num_frames 121 --steps 10 --lora_alpha 1.0
```

### 2026-06-06 BDD Lower-Alpha Comparison

Alpha `0.5` and `0.7` were generated on the exact same 12 BDD rows and directly
reviewed.

- Both runs completed 12/12 successfully.
- Neither lower alpha materially improves the overall visual failure rate.
- Persistent failures across all tested alpha values:
  `video7040`, `video1407`, `video3237`, `video6180`, and `video6`.
- Alpha `0.5` helps some phase-2 samples but damages previously clean
  `video2149`.
- Alpha `0.7` is visually close to alpha `1.0`.

Decision:

- Do not promote alpha `0.5`, `0.7`, or `1.0` as the sole full-test generator.
- Mixed-view validation is still required because the fixed-camera and
  vehicle/IP distributions may behave differently.
- For BDD, the next useful experiment is multi-seed generation plus a
  hallucination/freeze/drift reranker, not more alpha-only tuning.

### 2026-06-06 WTS Mixed-View Review

Alpha `0.7` validation completed and was directly reviewed:

```text
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_vehicle4_a07
/workspace/experiments/wan_ticft/diffsynth_lora_balanced_500_121f_fixed4_a07
```

Vehicle/IP:

- 4/4 technically successful.
- One coherent sample, one weak-motion sample, one severe hallucination, and
  one near-freeze.
- Do not use the LoRA as the primary vehicle/IP generator.

Fixed-overhead:

- 4/4 technically successful.
- Background and camera geometry are highly stable.
- One strong moving-vehicle sample; the remaining samples show limited motion
  or near-freeze.
- Keep only as a pixel-preserving candidate pending quantitative comparison
  against repeat-last.

Updated system decision:

- BDD: official Wan multi-seed candidates plus reranking.
- Vehicle/IP: official Wan multi-seed candidates plus reranking.
- Fixed-overhead: compare repeat-last, LoRA, and optional actor-motion
  candidate; strongly weight background consistency.
- Stop spending compute on DiffSynth alpha sweeps.

### 2026-06-06 WTS Phases 1-4 Final LoRA Decision

Expanded validation completed:

- Fixed-overhead phases 1-4: 20/20 successful and directly reviewed.
- Vehicle/IP phases 1-4: 20/20 successful and directly reviewed.

Findings:

- Fixed-overhead preserves the background exceptionally well, but many outputs
  are effectively frozen. Later phases sometimes produce useful actor motion,
  but also late-appearing or enlarged vehicles.
- Vehicle/IP has frequent freeze, weak motion, actor persistence, actor
  enlargement, and late hallucinated vehicles.

Final decision:

- DiffSynth LoRA is not the primary generator for any view.
- Do not run more broad LoRA validation or alpha sweeps.
- Keep LoRA only as an optional fixed-overhead candidate.
- Main development now moves to history-aware candidate generation and
  view-specific reranking.

### Validation Candidate Evaluator

Added:

```text
/workspace/wan_ticft/evaluation/evaluate_video_reports.py
```

It compares generated-video reports against the real target frames referenced
by a validation manifest. It supports multiple candidate reports and an
optional repeat-last baseline. Use it to verify that multi-seed generation
actually improves PSNR/SSIM before building a blind test-set reranker.
