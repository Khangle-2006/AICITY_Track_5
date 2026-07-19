# StableVSR Depth Gate

Purpose: test temporal diffusion video super-resolution/detail synthesis on the
current `depth_a75` scaffold. This targets the main leaderboard gap:
FID/FVD/LPIPS, while using `depth_a75` to preserve geometry.

References:

- StableVSR GitHub: `https://github.com/claudiom4sir/StableVSR`
- StableVSR model card: `https://huggingface.co/claudiom4sir/StableVSR`

Setup:

```bash
cd /workspace
conda activate wan22
bash /workspace/wan_ticft/evaluation/setup_stablevsr_env.sh
```

Gate:

```bash
cd /workspace
conda activate wan22
LIMIT=4 STEPS=20 CUDA_VISIBLE_DEVICES=2 \
  bash /workspace/wan_ticft/evaluation/run_stablevsr_depth_bdd4_gate.sh
```

If it OOMs, reduce steps first:

```bash
LIMIT=2 STEPS=10 CUDA_VISIBLE_DEVICES=2 \
  bash /workspace/wan_ticft/evaluation/run_stablevsr_depth_bdd4_gate.sh
```

Outputs:

```text
/workspace/experiments/wan_ticft/stablevsr_depth_a75_bdd4_s20_eval.csv
/workspace/experiments/wan_ticft/stablevsr_depth_a75_bdd4_s20_proxy.csv
/workspace/experiments/wan_ticft/stablevsr_depth_a75_bdd4_s20_compare_sheets
```

The gate script now includes LR dimensions in new output names, for example:

```text
/workspace/experiments/wan_ticft/stablevsr_depth_a75_bdd4_320x180_s20_*
```

Optional less-aggressive LR variant:

```bash
LIMIT=4 STEPS=10 LR_WIDTH=480 LR_HEIGHT=270 CUDA_VISIBLE_DEVICES=2 \
  bash /workspace/wan_ticft/evaluation/run_stablevsr_depth_bdd4_gate.sh
```

Decision:

- Continue only if FID proxy/visual detail improves without obvious temporal
  flicker or object identity changes.
- If FID/FVD worsens or the output becomes over-sharpened/hallucinated, close
  StableVSR and try a more conservative temporal detailer or residual diffusion
  trained on Track 5.
