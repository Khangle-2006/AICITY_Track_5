#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft
THIRD=/workspace/third_party

targets=(
  # Cosmos-Predict2.5 is closed after gated checkpoint download returned 403.
  "$ROOT/cosmos25_bdd_smoke4"
  "$ROOT/cosmos25_diffusers_env"
  "$ROOT/cosmos25_diffusers_smoke17"
  "$ROOT/cosmos25_hf_cache"
  "$THIRD/cosmos-predict2.5"

  # Epona is fully closed.
  "$ROOT/epona_env"
  "$ROOT/epona_pretrained"
  "$ROOT/epona_one_frame_gate4"
  "$ROOT/epona_one_frame_smoke"
  "$ROOT/epona_orientation_correction"
  "$ROOT/epona_rollout4_gate"
  "$THIRD/Epona"

  # MemFlow-P and VDPM are fully closed.
  "$THIRD/MemFlow"
  "$THIRD/vdpm"
  "$ROOT/vdpm_env"
  "$ROOT/vdpm_bdd_smoke4"
  "$ROOT/vdpm_bdd_smoke4_motion"

  # LTX LoRA training was closed. Keep the deployed LTX-Video repo and ltxv_env.
  "$ROOT/ltx_lora_trainer_env"
  "$THIRD/LTX-Video-Trainer"
  "$ROOT/ltx_lora_bdd_smoke20"
  "$ROOT/ltx_lora_bdd_smoke20_output"
  "$ROOT/ltx_lora_bdd_smoke_eval4"
  "$ROOT/ltx_lora_bdd_smoke_eval4_lora_sheets"
  "$ROOT/ltx_ic_lora_bdd_smoke20"
  "$ROOT/ltx_ic_lora_bdd_smoke20_output"
  "$ROOT/ltx_ic_lora_bdd_smoke_eval5"
  "$ROOT/ltx_ic_lora_bdd_smoke_eval5_ic_lora_sheets"

  # Learned future-flow experiments were closed.
  "$ROOT/bdd_future_flow_calibration_gate"
  "$ROOT/bdd_future_flow_predictor_smoke"
  "$ROOT/bdd_layered_residual_flow_smoke"

  # Closed LTX history, prompt, and future-anchor tuning outputs.
  "$ROOT/ltxv_bdd_adaptive_history_heldout4"
  "$ROOT/ltxv_bdd_adaptive_history_heldout4_sheets"
  "$ROOT/ltxv_bdd_history17_smoke4"
  "$ROOT/ltxv_bdd_history17_smoke4_sheets"
  "$ROOT/ltxv_bdd_history25_correction4"
  "$ROOT/ltxv_bdd_history25_correction4_sheets"
  "$ROOT/ltxv_bdd_manifest_prompt_smoke4"
  "$ROOT/ltxv_bdd_manifest_prompt_smoke4_sheets"
  "$ROOT/ltxv_bdd_behavior_prompt_correction4"
  "$ROOT/ltxv_bdd_behavior_prompt_correction4_sheets"
  "$ROOT/ltxv_bdd_future_anchor4_gate"
  "$ROOT/ltxv_bdd_future_anchor4_gate_sheets"
  "$ROOT/ltxv_bdd25_transition16_color_anchor"
  "$ROOT/ltxv_bdd25_transition16_smoothstep"
)

echo "=== Closed-experiment cleanup ==="
echo "Preserving Wan2.2, DiffSynth-Studio, LTX-Video, ltxv_env,"
echo "transition16 candidates, manifests, validation frames, CSVs, and source code."
echo
df -h /workspace

existing=()
for target in "${targets[@]}"; do
  case "$target" in
    "$ROOT"/*|"$THIRD"/*) ;;
    *)
      echo "Unsafe target outside allowed roots: $target" >&2
      exit 2
      ;;
  esac
  if [[ -e "$target" ]]; then
    existing+=("$target")
  fi
done

echo
echo "Deleting ${#existing[@]} verified closed paths:"
if ((${#existing[@]})); then
  du -sh "${existing[@]}" 2>/dev/null || true
fi

for target in "${existing[@]}"; do
  echo "remove $target"
  rm -rf -- "$target"
done

echo
echo "After cleanup:"
df -h /workspace
echo "Remaining requested paths:"
remaining=0
for target in "${targets[@]}"; do
  if [[ -e "$target" ]]; then
    echo "$target"
    remaining=$((remaining + 1))
  fi
done
echo "remaining=$remaining"
