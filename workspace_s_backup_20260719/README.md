# Workspace S Backup 2026-07-19

This folder is a lightweight preservation snapshot of the AI City Track 5 work that was still present on the `S:` drive before the machine reinstall.

It is intended for code and experiment traceability, not as a full runnable dataset mirror.

## Included

- `wan_ticft/`: experiment and inference code from `S:\wan_ticft`, excluding datasets, caches, model weights, videos, images, and archives.
- `baseline/`: small baseline scripts from `S:\baseline`.
- `docs/`: small documentation files from `S:\docs`.
- `root_scripts/`: top-level launch scripts, handoff notes, and patch files from `S:\`.
- `reports/`: candidate report CSV files from `S:\experiments\wan_ticft`.
- `manifests/`: file and directory listings produced before backup to help locate larger artifacts.

## Excluded

- Raw datasets under `S:\data` and dataset symlinks.
- Model caches, checkpoints, weights, and environment folders.
- Generated frame folders and full submission directories.
- Large videos, images, and zip archives.

Large visual artifacts and submission archives were preserved separately on the local backup drive under:

```text
G:\AICITY_Backup_20260719
```

