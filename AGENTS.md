# Repository Guidelines

## Project Structure & Module Organization

This workspace contains the AI City Challenge Track 5 forecasting baseline. The main codebase is `wts_baseline/`; Docker support lives in `aicity-track5_docker/`, and project notes are kept in root Markdown/docx files.

Inside `wts_baseline/`, Python package code is under `src/`: datasets in `src/datasets/`, losses in `src/losses/`, models in `src/models/`, and end-to-end video pipelines in `src/pipelines/`. Operational entrypoints are under `scripts/prepare/`, `scripts/train/`, `scripts/infer/`, and `scripts/eval/`. Shell wrappers such as `run_training_v2.sh` and `run_sota_full_pipeline.sh` are the preferred reproducible commands. Data, checkpoints, predictions, and logs are stored in `data/`, `checkpoints*/`, `prediction*/`, and `logs*/`.

## Build, Test, and Development Commands

Run commands from `wts_baseline/` unless noted:

```bash
export PYTHONPATH=/workspace/wts_baseline:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- `./run_training_v2.sh`: trains the V2 SD UNet + AnimateDiff baseline.
- `bash run_sota_full_pipeline.sh 0 10 10 5`: runs SOTA preparation, staged training, inference, and evaluation on GPU 0 with explicit epoch counts.
- `./run_validation_eval.sh`: generates validation predictions and computes metrics.
- `python3 scripts/infer/infer.py --manifest data/manifests/val.jsonl --checkpoint checkpoints_v2/best.pt --output_dir /tmp/wts_smoke --use_sd_unet --max_samples 1`: quick inference smoke test.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, `snake_case` for functions, modules, variables, and script filenames, and `PascalCase` for model or dataset classes. Keep imports grouped standard library, third-party, then local `src.*` imports. Prefer explicit argparse flags for experiment settings and keep generated artifacts out of `src/` and `scripts/`.

## Testing Guidelines

There is no formal test suite in this workspace. Validate changes with the smallest relevant smoke run first, typically `--max_samples 1`, then run `scripts/eval/compute_metrics.py` or `scripts/eval/eval_sota.py` on validation outputs when behavior changes. For data preparation changes, inspect the resulting `data/manifests/*.jsonl` before training.

## Commit & Pull Request Guidelines

The root workspace does not include Git history, so use clear imperative commit messages such as `Add WAN inference smoke command` or `Fix SOTA manifest parsing`. Pull requests should describe the affected pipeline, list commands run, identify checkpoints or datasets used, and include metric changes or sample output paths for inference-facing work.

## Agent-Specific Instructions

See `wts_baseline/AGENTS.md` for detailed pipeline notes, GPU setup, data layout, and architecture caveats before editing model, inference, or evaluation code.
