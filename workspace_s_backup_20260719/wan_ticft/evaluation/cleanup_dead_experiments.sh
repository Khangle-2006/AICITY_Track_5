#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments

echo "=== Cleanup dead Track 5 experiments ==="
echo "root=$ROOT"
echo "before:"
df -h /workspace || true

delete_paths=(
  "$ROOT/hybrid_cnn_ltxmix45_wts_guard015_candidate"
  "$ROOT/hybrid_cnn_selector_region_wts_submission"
  "$ROOT/hybrid_cnn_selector_routed_wts_submission"
  "$ROOT/hybrid_ltx_minimal_lowfreq_wts_domain_routed_candidate"
  "$ROOT/hybrid_ltx_phase0_transition32_wts_domain_routed_candidate"
  "$ROOT/hybrid_ltx_region_wts_submission"
  "$ROOT/hybrid_ltx_transition16_region_wts_submission"
  "$ROOT/hybrid_ltx_transition16_wts_domain_routed_candidate"
  "$ROOT/hybrid_ltx_transition16_wts_overhead_object_layer_candidate"
  "$ROOT/hybrid_ltx_transition16_wts_vehicle_guard25_candidate"
  "$ROOT/hybrid_wan_region_fixed_submission"
  "$ROOT/hybrid_wan_region_wts_submission"
  "$ROOT/region_norm_medium_submission"
  "$ROOT/region_norm_medium_test_smoke_bdd"
  "$ROOT/region_norm_medium_test_smoke_bdd_sheets"
  "$ROOT/wan22_submission_ti2v_121f_s10"

  "$ROOT/wan_ticft/cosmos3_access_test"
  "$ROOT/wan_ticft/cosmos3_bdd_v2v_smoke4"
  "$ROOT/wan_ticft/cosmos3_bdd_v2v_smoke4_debug_sheets"
  "$ROOT/wan_ticft/cosmos3_smoke"
  "$ROOT/wan_ticft/cosmos3_background_hybrid_smoke4"
  "$ROOT/wan_ticft/cosmos3_background_hybrid_smoke4_bg_s31_a15_sheets"
  "$ROOT/wan_ticft/cosmos3_background_hybrid_smoke4_bg_s31_a25_sheets"
  "$ROOT/wan_ticft/cosmos3_background_hybrid_smoke4_bg_s45_a20_sheets"
  "$ROOT/wan_ticft/cosmos3_background_masked_hybrid_smoke4"
  "$ROOT/wan_ticft/cosmos3_background_masked_hybrid_smoke4_bgmask_s31_a25_m03_e40_sheets"
  "$ROOT/wan_ticft/cosmos3_background_masked_hybrid_smoke4_bgmask_s31_a30_m04_e50_sheets"
  "$ROOT/wan_ticft/cosmos3_background_masked_hybrid_smoke4_bgmask_s45_a25_m03_e40_sheets"
  "$ROOT/wan_ticft/cosmos3_bdd_anchor_correction4"
  "$ROOT/wan_ticft/cosmos3_bdd_anchor_correction4_debug_sheets"

  "$ROOT/wan_ticft/diffsynth_lora_balanced_500_121f_lr5e5"
  "$ROOT/wan_ticft/ltx_flow_bridge_gate"
  "$ROOT/wan_ticft/ltx_masked_transition_gate"
  "$ROOT/wan_ticft/ltx_region_lowfreq_s16_a50_val24"
  "$ROOT/wan_ticft/ltx_region_lowfreq_s16_a50_val24_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_dense6_eval"
  "$ROOT/wan_ticft/ltxv_bdd_gate_video1170"
  "$ROOT/wan_ticft/ltxv_bdd_gate_video1170_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_minimal_prompt_lowfreq_s16_a50_smoke4"
  "$ROOT/wan_ticft/ltxv_bdd_minimal_prompt_lowfreq_s16_a50_smoke4_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_minimal_prompt_smoke4"
  "$ROOT/wan_ticft/ltxv_bdd_minimal_prompt_smoke4_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_mixed4_gate"
  "$ROOT/wan_ticft/ltxv_bdd_mixed4_gate_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_smoke4"
  "$ROOT/wan_ticft/ltxv_bdd_test_full"
  "$ROOT/wan_ticft/ltxv_bdd_test_full_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_test_minimal_lowfreq_s16_a50"
  "$ROOT/wan_ticft/ltxv_bdd_test_minimal_prompt_full"
  "$ROOT/wan_ticft/ltxv_bdd_test_phase0_transition32_route"
  "$ROOT/wan_ticft/ltxv_bdd_test_smoke4"
  "$ROOT/wan_ticft/ltxv_bdd_test_smoke4_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_test_transition16_sheets"
  "$ROOT/wan_ticft/ltxv_bdd_test_transition32"
  "$ROOT/wan_ticft/ltxv_bdd_val2_diagnostic"
  "$ROOT/wan_ticft/ltxv_bdd_val2_gate"
  "$ROOT/wan_ticft/ltxv_bdd_val2_gate_sheets"
  "$ROOT/wan_ticft/ltxv_bdd25_gate_sheets"
  "$ROOT/wan_ticft/ltxv_bdd25_transition_duration_gate"
  "$ROOT/wan_ticft/ltxv_bdd25_transitions"
  "$ROOT/wan_ticft/ltxv_vehicle4_gate"
  "$ROOT/wan_ticft/ltxv_vehicle4_gate_sheets"

  "$ROOT/wan_ticft/region_norm_medium_wts_audit"
  "$ROOT/wan_ticft/region_norm_medium_wts_audit_sheets"
  "$ROOT/wan_ticft/region_norm_routed_wts_test_sheets"
  "$ROOT/wan_ticft/vace_region_bdd_gate_video1170"
  "$ROOT/wan_ticft/vace_region_bdd_gate_video1170_sheets"
  "$ROOT/wan_ticft/wan22_bdd25_seed101_s10"
  "$ROOT/wan_ticft/wan22_ti2v_test_121f_s10"
  "$ROOT/wan_ticft/wan22_wts100_s10"

  "$ROOT/wan_ticft/wts_actor_guard_gate20"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa_guard25_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa_guard50_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa_long24_guard50_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa_region_long24_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa_region_medium_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_overhead_or_fixed_camera_visual_qa_target_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa_guard25_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa_guard50_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa_long24_guard50_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa_region_long24_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa_region_medium_sheets"
  "$ROOT/wan_ticft/wts_actor_guard_vehicle_or_ip_camera_visual_qa_target_sheets"
  "$ROOT/wan_ticft/wts_distribution_gate20"
  "$ROOT/wan_ticft/wts_object_layer_full100"
  "$ROOT/wan_ticft/wts_object_layer_full90"
  "$ROOT/wan_ticft/wts_object_layer_gate20"
  "$ROOT/wan_ticft/wts_object_layer_guarded_full90"
  "$ROOT/wan_ticft/wts_object_layer_small_gate20"
  "$ROOT/wan_ticft/wts_overhead_long24_vehicle_guard25_test"
  "$ROOT/wan_ticft/wts_overhead_long24_vehicle_guard25_test_sheets"
  "$ROOT/wan_ticft/wts_overhead_long24_vehicle_medium_test_minimal_lowfreq"
  "$ROOT/wan_ticft/wts_overhead_long24_vehicle_medium_test_phase0_route32"
  "$ROOT/wan_ticft/wts_overhead_long24_visual_qa"
  "$ROOT/wan_ticft/wts_overhead_long24_visual_qa_region_long24_sheets"
  "$ROOT/wan_ticft/wts_overhead_long24_visual_qa_region_medium_sheets"
  "$ROOT/wan_ticft/wts_overhead_long24_visual_qa_target_sheets"
  "$ROOT/wan_ticft/wts_texture_postprocess_gate20"
)

deleted=0
missing=0
for path in "${delete_paths[@]}"; do
  case "$path" in
    "$ROOT"/*) ;;
    *) echo "Refusing to delete outside $ROOT: $path" >&2; exit 2 ;;
  esac
  if [[ -e "$path" ]]; then
    echo "delete $path"
    rm -rf --one-file-system "$path"
    deleted=$((deleted + 1))
  else
    missing=$((missing + 1))
  fi
done

echo "deleted=$deleted missing=$missing"
echo "after:"
df -h /workspace || true

echo
echo "Kept important active/fallback dirs:"
cat <<EOF
  $ROOT/index
  $ROOT/hf_cache
  $ROOT/hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate
  $ROOT/hybrid_cnn_ltxmix65_wts_guard015_candidate
  $ROOT/hybrid_cnn65_cosmos_bg_s31_a25_wts_guard015_candidate
  $ROOT/wan_ticft/cosmos3_direct
  $ROOT/wan_ticft/cosmos3_bdd_test_anchor44
  $ROOT/wan_ticft/cosmos3_bdd_anchor_gate24
  $ROOT/wan_ticft/cosmos3_background_hybrid_gate24
  $ROOT/wan_ticft/cnn_ltx_region_selector_bdd_test
  $ROOT/wan_ticft/cnn_ltxmix65_bdd_test
  $ROOT/wan_ticft/ltxv_bdd_test_transition16
  $ROOT/wan_ticft/wts_guard015_test
  $ROOT/wan_ticft/wts_guard_current_components
  $ROOT/wan_ticft/wts_guard_actor_components
  $ROOT/wan_ticft/wts_overhead_object_layer_vehicle_medium_test
EOF
