#!/usr/bin/env bash
set -euo pipefail

safe_rm() {
  local target="$1"
  local resolved
  resolved="$(realpath -m "$target")"
  case "$resolved" in
    /workspace/experiments/*)
      ;;
    /workspace/third_party/ATI)
      ;;
    /workspace/third_party/cosmos-predict2.5)
      ;;
    /workspace/models/Wan-AI/Wan2.1-VACE-1.3B/._____temp)
      ;;
    /workspace/models/Wan-AI/Wan2.1-VACE-1.3B)
      ;;
    /workspace/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth)
      ;;
    *)
      echo "Refusing unsafe target: $resolved" >&2
      exit 1
      ;;
  esac
  if [[ -e "$resolved" || -L "$resolved" ]]; then
    echo "Deleting $resolved"
    rm -rf -- "$resolved"
  fi
}

# Materialized data and duplicate frame exports. Source data and manifests remain.
for name in \
  frames_121f_smoke \
  frames_balanced_121f \
  frames_balanced_300 \
  frames_balanced_smoke \
  frames_val_121f_eval \
  frames_val_bdd12_matched_121f \
  frames_val_bdd12_matched_121f_ffmpeg \
  frames_val_bdd12_matched_121f_v2 \
  wan22_ti2v_test_121f_s10_frames \
  wan22_ti2v_test_121f_s10_submission_sheets
do
  safe_rm "/workspace/experiments/wan_ticft/$name"
done

# Superseded baseline and refiner experiments.
for name in \
  baseline_e2e_flow \
  baseline_e2e_smoke \
  baseline_e2e_view_hybrid \
  baseline_submission_flow \
  baseline_submission_view_hybrid \
  c1_history_fusion_test \
  c1_history_fusion_val \
  c1_history_fusion_val_all_gap4 \
  c1_history_fusion_val_all_gap8 \
  c1_history_fusion_val_fixed \
  c1_history_fusion_val_gap4 \
  c1_history_fusion_val_gap8 \
  c1_history_fusion_val_residual \
  c2_motion_prior_val_all_gap4 \
  c2_motion_prior_val_all_gap8 \
  c2_motion_prior_val_gap4 \
  c3_caption_decomposition_val \
  c3_event_decomposition_val \
  refiner_bdd_direct_h1_120_v2_smoke \
  refiner_contact_sheets \
  refiner_contact_sheets_rollout_smoke \
  refiner_e2e_anchor_rollout_a035_smoke \
  refiner_e2e_mixed_h1_16 \
  refiner_e2e_mixed_h1_16_videos_30fps \
  refiner_e2e_rollout_smoke \
  refiner_e2e_smoke \
  refiner_gap4_v1 \
  refiner_gap8_v1 \
  refiner_mixed_h1_16_v1 \
  refiner_smoke_gap4 \
  refiner_submission_mixed_h1_16 \
  videos_debug \
  visual_debug
do
  safe_rm "/workspace/experiments/$name"
done

# Region outputs duplicated by retained submission folders.
for name in \
  region_norm_medium_test \
  region_norm_medium_test_sheets \
  region_norm_medium_test_smoke4 \
  region_norm_medium_test_smoke4_sheets \
  region_norm_medium_test_smoke_wts_overhead_or_fixed_camera \
  region_norm_medium_test_smoke_wts_overhead_or_fixed_camera_sheets \
  region_norm_medium_test_smoke_wts_vehicle_or_ip_camera \
  region_norm_medium_test_smoke_wts_vehicle_or_ip_camera_sheets
do
  safe_rm "/workspace/experiments/$name"
done

# Already rejected ATI branch and failed partial ModelScope cache.
safe_rm /workspace/third_party/ATI

# Rejected VACE and Cosmos-Predict2.5 gates.
safe_rm /workspace/experiments/wan_ticft/vace_region_bdd_gate_video1170
safe_rm /workspace/experiments/wan_ticft/vace_region_bdd_gate_video1170_sheets
safe_rm /workspace/models/Wan-AI/Wan2.1-VACE-1.3B
safe_rm /workspace/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
safe_rm /workspace/third_party/cosmos-predict2.5

echo
echo "Cleanup complete."
echo "Workspace usage:"
df -h /workspace
