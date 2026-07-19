#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
ENV_DIR="$ROOT/cosmos25_diffusers_env"
CACHE_DIR="$ROOT/skyreels_v2_hf_cache"
SOURCE_MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
REUSE_REPORT="$ROOT/skyreels_v2_df_v2v_bdd1_smoke/report.csv"
SKY_OUT="$ROOT/skyreels_v2_df_v2v_bdd4_gate"
OUT="$ROOT/motion_locked_texture_skyreels_bdd4_lossless"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Motion-locked SkyReels texture BDD4 gate ==="
echo "This generates only missing SkyReels donors, then runs a lossless texture gate."
echo "source_manifest=$SOURCE_MANIFEST"
echo "sky_out=$SKY_OUT"
echo "out=$OUT"
echo "gpu=$GPU_ID"

for path in "$SOURCE_MANIFEST" "$BASE_REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 2
  fi
done
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing env: $ENV_DIR" >&2
  exit 2
fi

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000
mkdir -p "$SKY_OUT/videos" "$CACHE_DIR"

cat > "$SKY_OUT/generate_skyreels_v2_df_v2v_bdd4.py" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingVideoToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image


ROOT = Path("/workspace/experiments/wan_ticft")
MANIFEST = ROOT / "ltxv_bdd25_gate/dense40_manifest.jsonl"
REUSE_REPORT = ROOT / "skyreels_v2_df_v2v_bdd1_smoke/report.csv"
OUT = ROOT / "skyreels_v2_df_v2v_bdd4_gate"
VIDEO_DIR = OUT / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with MANIFEST.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("view_type") == "bdd":
            rows.append(row)
        if len(rows) >= 4:
            break
if len(rows) != 4:
    raise RuntimeError(f"expected four BDD rows, found {len(rows)}")

reuse = {}
if REUSE_REPORT.exists():
    with REUSE_REPORT.open("r", encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            path = Path(item.get("video", ""))
            if item.get("status") == "ok" and path.exists():
                reuse[item["sample_id"]] = path

pending = [row for row in rows if row["sample_id"] not in reuse and not (VIDEO_DIR / f"{row['sample_id']}.mp4").exists()]
pipe = None
if pending:
    model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
    print(f"loading {model_id} for {len(pending)} missing donors", flush=True)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

prompt = (
    "Forward-facing dashcam driving video. Continue the same road scene into the near future. "
    "Preserve camera viewpoint, road geometry, lane layout, vehicles, pedestrians, signs, lighting, "
    "weather, dashboard, and object identities. Only physically plausible traffic motion changes."
)

report = []
for position, row in enumerate(rows, start=1):
    sample_id = row["sample_id"]
    if sample_id in reuse:
        output = reuse[sample_id]
        status_note = "reused_bdd1"
    else:
        output = VIDEO_DIR / f"{sample_id}.mp4"
        status_note = "reused_existing" if output.exists() else "generated"
        if not output.exists():
            condition_paths = (row.get("frames") or {}).get("condition") or []
            video = [load_image(path).convert("RGB") for path in condition_paths]
            frames = pipe(
                video=video,
                prompt=prompt,
                height=544,
                width=960,
                num_frames=97,
                base_num_frames=97,
                num_inference_steps=12,
                guidance_scale=5.0,
                ar_step=0,
                overlap_history=17,
                addnoise_condition=20,
                fps=16,
                generator=torch.Generator(device="cuda").manual_seed(20260620 + position),
            ).frames[0]
            export_to_video(frames, str(output), fps=16, quality=8)
    report.append(
        {
            "sample_id": sample_id,
            "video": str(output),
            "status": "ok",
            "future_frames": 101,
            "notes": status_note,
        }
    )
    print(f"[{position}/4] {sample_id} | {status_note} | {output}", flush=True)

with (OUT / "report.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)
with (OUT / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH=/workspace \
HF_HUB_CACHE="$CACHE_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 180m "$ENV_DIR/bin/python" "$SKY_OUT/generate_skyreels_v2_df_v2v_bdd4.py"

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.materialize_motion_locked_texture \
  --manifest "$SKY_OUT/selected_manifest.jsonl" \
  --base_report "$BASE_REPORT" \
  --skyreels_report "$SKY_OUT/report.csv" \
  --out_root "$OUT" \
  --limit 4 \
  --variants lossless_control,sky_yhf_a25,sky_yhf_a35,condition_yhf_a25 \
  --clean

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    lossless_control="$OUT/lossless_control/report.csv" \
    sky_yhf_a25="$OUT/sky_yhf_a25/report.csv" \
    sky_yhf_a35="$OUT/sky_yhf_a35/report.csv" \
    condition_yhf_a25="$OUT/condition_yhf_a25/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report lossless_control="$OUT/lossless_control/report.csv" \
  --report sky_yhf_a25="$OUT/sky_yhf_a25/report.csv" \
  --report sky_yhf_a35="$OUT/sky_yhf_a35/report.csv" \
  --report condition_yhf_a25="$OUT/condition_yhf_a25/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 2

mapfile -t SAMPLE_IDS < <(/opt/conda/envs/wan22/bin/python - <<PY
import json
for line in open("$OUT/selected_manifest.jsonl", encoding="utf-8"):
    if line.strip():
        print(json.loads(line)["sample_id"])
PY
)
SHEET_ARGS=()
for sample_id in "${SAMPLE_IDS[@]}"; do
  SHEET_ARGS+=(--sample_id "$sample_id")
done

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    lossless_control="$OUT/lossless_control/report.csv" \
    sky_yhf_a25="$OUT/sky_yhf_a25/report.csv" \
    sky_yhf_a35="$OUT/sky_yhf_a35/report.csv" \
    condition_yhf_a25="$OUT/condition_yhf_a25/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "BDD4 outputs:"
echo "  eval=${OUT}_eval.csv"
echo "  proxy=${OUT}_proxy.csv"
echo "  sheets=${OUT}_compare_sheets"
echo
echo "Gate rule:"
echo "  - lossless_control must match cnn_ltxmix65."
echo "  - Continue only if a SkyReels variant improves aggregate FVD, keeps FID increase below 0.15,"
echo "    loses no more than 0.10 PSNR / 0.002 SSIM, and has no halos or texture crawl."
