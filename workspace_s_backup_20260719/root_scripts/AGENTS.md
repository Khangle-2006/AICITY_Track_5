# AI City Track 5 Workspace Instructions

This workspace develops a Generative Traffic Video Forecasting submission.
Read these files before making changes or launching jobs:

- `/workspace/docs/track5_status.md`
- `/workspace/docs/visual_review.md`
- `/workspace/docs/experiment_log.md`

## Working Rules

- Inspect existing code, reports, manifests, and checkpoints before rerunning work.
- Keep modules separated by responsibility:
  - `/workspace/baseline`: classical/refiner baselines, diagnostics, submission tools.
  - `/workspace/wan_ticft/data`: manifest and frame preparation.
  - `/workspace/wan_ticft/training`: training data and backend utilities.
  - `/workspace/wan_ticft/inference`: Wan and DiffSynth inference.
- Do not move unrelated code into one generic pipeline directory.
- Do not delete or overwrite existing experiments, checkpoints, models, caches, or
  `third_party` files without explicit approval.
- Use new, descriptive experiment output directories.
- Run smoke tests before full jobs. Verify reports and create contact sheets.
- Update `/workspace/docs/experiment_log.md` after every meaningful experiment.

## Server Safety

- This is a shared server. Do not modify system CUDA, drivers, or global packages.
- Use conda environments:
  - `wan22`: Wan2.2 and DiffSynth work.
  - `codex-cli`: Codex CLI only.
- GPUs:
  - GPU 0: NVIDIA A100 40GB.
  - GPU 1: NVIDIA A100 40GB.
  - GPU 2/3: NVIDIA RTX A6000 48GB and may be occupied.
- Check `nvidia-smi` before long jobs.
- Prefer one GPU for training unless multi-GPU behavior has been tested.
- Never use destructive git or filesystem commands against `/workspace`.

## Submission Contract

- There are 71 test samples.
- Each output sample folder must match the expected test sample output name.
- Generate exactly `target_N` future frames for each sample.
- Frame resolution must be `1280x720`.
- Submission frame names should be `0.png`, `1.png`, ..., not zero-padded.
- The accepted archive layout is expected to have sample folders directly at the
  ZIP root, without an extra `prediction/` directory.

## Current Priorities

1. Validate the newly trained balanced 500-sample, 121-frame DiffSynth LoRA.
2. Compare it against the submitted official Wan2.2 zero-shot baseline.
3. Build multi-seed official Wan generation and candidate reranking.
4. Add view-specific handling, especially background preservation for fixed
   overhead cameras.
5. Only consider a full test generation after mixed validation and visual QA.
