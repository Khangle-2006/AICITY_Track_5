# Prompt For New Chat

Use this prompt when starting a new Codex/chat session.

---

You are taking over an AI City Track 5 video prediction project. Work directly in the shared server workspace mounted as `S:` on Windows:

- Repo: `/workspace/wan_ticft` (`S:\wan_ticft`)
- Experiments: `/workspace/experiments` (`S:\experiments`)
- Index: `/workspace/experiments/index`

First read these files:

1. `/workspace/wan_ticft/PROJECT_HANDOFF.md`
2. `/workspace/wan_ticft/EXPERIMENT_LOG.md`
3. `/workspace/wan_ticft/NEXT_CHAT_PROMPT.md`

Then inspect the relevant experiment outputs and contact sheets:

- `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24_eval.csv`
- `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24_proxy.csv`
- `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24_sheets`
- `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24_eval.csv` if it exists
- `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24_proxy.csv` if it exists
- `/workspace/experiments/wan_ticft/temporal_latent_feature_train100_gate24_sheets` if it exists
- `/workspace/experiments/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate`
- `/workspace/experiments/hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate`

The current goal is not minor tweaking. We need a real improvement toward a much higher score. Current best public is about 69.14, far below the desired 80. The user wants bounded experiments: if a direction is bad after a smoke/correction, close it; if a direction has real potential, exploit it seriously.

Current best known public candidate:

```text
hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate
Score 69.1436
PSNR 18.5786
SSIM 0.6222
LPIPS 0.3295
CLIP 0.9390
FID 48.5029
FVD 31.2674
```

Current promising direction:

Temporal latent feature refiner. It improved BDD pixel metrics strongly and did not catastrophically worsen FVD on heldout8.

Heldout8:

```text
cnn_ltxmix65              PSNR 17.8461  SSIM 0.7193  FID 7.4616  FVD 25.8754
temporal_feature_single   PSNR 18.1969  SSIM 0.7237  FID 7.5998  FVD 25.6957
```

BDD24 val-overfit:

```text
cnn_ltxmix65        PSNR 18.2129  SSIM 0.7877  FID 4.9677  FVD 22.2935
temporal_feature    PSNR 18.7337  SSIM 0.7990  FID 5.5718  FVD 23.1755
```

This is worth one public submission, but the real path is train-on-train:

```bash
cd /workspace
conda activate wan22
CUDA_VISIBLE_DEVICES=3 bash /workspace/wan_ticft/evaluation/run_temporal_latent_feature_train100_gate24.sh
```

That script trains on 100 BDD train samples and validates on BDD24. If it passes, scale to 300 BDD train samples. If it fails, do not blindly tune losses; the issue may be mismatch between train base (region-motion) and eval/test base (`cnn_ltxmix65`).

Current verified temporal candidate folder:

```text
/workspace/experiments/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate
```

It was verified as:

```text
71 rows / 5194 frames
```

Robocopy command:

```cmd
robocopy S:\experiments\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate /E /MT:16 /R:2 /W:2
```

Zip locally:

```cmd
powershell Compress-Archive -Path G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate\* -DestinationPath G:\hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate.zip -Force
```

Do not repeat closed directions unless new evidence appears:

- Alignment/color audit
- Multi-history flow
- Epona rollout/keyframe hybrid
- Cosmos3 current image-anchor/5-frame route
- Wan2.2-TI2V and VACE refiner
- RGB output adapters
- Feature/distribution/adversarial adapters
- LTX prompt/history/STG/transition tuning
- Simple rerank/router
- WTS texture/unsharp/CLAHE
- Hand-engineered WTS object guard beyond current overhead-object candidate

User preferences:

- Be direct and pragmatic.
- Do not overfit public/test composition.
- Do not waste time on tiny sweeps unless a gate passed.
- If something is bad, close it.
- If a method has promise, improve it based on engineering judgment.
- Do not zip on server; user wants robocopy then local zip.
- If user says they copied error to `output.txt`, ask for exact path or search carefully; do not hallucinate file contents.

Security:

- The user pasted a Hugging Face token earlier. Do not print or store it. Use environment variables only.

Immediate next work:

1. Do not rerun `run_temporal_latent_feature_train100_gate24.sh` as-is. The gate
   was run after repairing missing frame PNGs and failed:
   `temporal_train100` PSNR `18.0991` vs `cnn_ltxmix65` `18.2129`, SSIM
   `0.7852` vs `0.7877`, FVD proxy `22.5984` vs `22.2935`.
2. Do not scale this region-motion train base to train300.
3. If continuing temporal refiner work, first create a BDD train base close to
   eval/test `cnn_ltxmix65`, then rerun the same BDD24 gate.
4. Only after a matched-base BDD pass, make a WTS-specific train/val refiner.
   Do not assume BDD refiner helps WTS.

---
