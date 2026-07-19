# Wan/TIC-FT Training

Current files:

- `manifest_dataset.py`: JSONL loader and frame-path validation.
- `layout.py`: TIC-FT condition/buffer/target layout helpers.
- `train_ticft.py`: dry-run training entrypoint and future Wan backend hook.

Dry-run:

```bash
python -m wan_ticft.training.train_ticft \
  --train_jsonl /workspace/experiments/wan_ticft/manifests/train_ticft_balanced_300_frames.jsonl \
  --val_jsonl /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --out_dir /workspace/experiments/wan_ticft/train_dryrun_300 \
  --backend dryrun \
  --strict
```

The future Wan backend must:

1. Load JSONL rows.
2. Encode condition and target frames with the Wan VAE.
3. Build `condition frames + noised condition buffer frames + noisy target frames`.
4. Freeze the Wan backbone for first experiments.
5. Train LoRA or spatial/temporal adapters.
6. Compute denoising loss only over the target frame slice.
