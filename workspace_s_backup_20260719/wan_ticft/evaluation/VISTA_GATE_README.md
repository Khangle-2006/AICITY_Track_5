# Vista BDD Gate

Purpose: test whether Vista, a driving-specific world model, can improve BDD
realism/FID/FVD enough to justify deeper integration into Track 5.

This first gate is deliberately small and conservative:

- uses Vista official `IMG` mode;
- conditions on the last manifest condition frame only;
- writes a normal Track-5-style video report so existing pixel/proxy/contact
  sheet tooling can score it;
- does not build a submission.

Official references used:

- Vista repository: `https://github.com/OpenDriveLab/Vista`
- Vista sampling docs: `https://github.com/OpenDriveLab/Vista/blob/main/docs/SAMPLING.md`
- Vista model weights: `https://huggingface.co/OpenDriveLab/Vista`

Setup:

```bash
cd /workspace
conda activate wan22
bash /workspace/wan_ticft/evaluation/setup_vista_env.sh
```

Smoke gate:

```bash
cd /workspace
conda activate wan22
LIMIT=4 N_STEPS=30 CUDA_VISIBLE_DEVICES=2 \
  bash /workspace/wan_ticft/evaluation/run_vista_img_bdd4_gate.sh
```

Outputs:

```text
/workspace/experiments/wan_ticft/vista_img_bdd4_s30_eval.csv
/workspace/experiments/wan_ticft/vista_img_bdd4_s30_proxy.csv
/workspace/experiments/wan_ticft/vista_img_bdd4_s30_compare_sheets
```

Gate decision:

- Continue only if `vista_img` preserves scene/road/actor identity and improves
  FID/FVD proxy versus `ltx2b_base` and preferably moves toward `depth_a75`.
- If pixel collapses and FID/FVD is not clearly better, close raw Vista IMG and
  move to temporal VSR/detailing on `depth_a75`.
- If raw Vista has good FID/FVD but poor pixel alignment, the next step is not
  submission; build a hybrid route or deeper adapter that uses `depth_a75` as
  scaffold and Vista only for texture/dynamic residual.

