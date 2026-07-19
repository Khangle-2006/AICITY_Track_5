#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments
WAN="$ROOT/wan_ticft"

echo "=== Closed Track 5 cleanup 2026-06-20 ==="
df -h /workspace

targets=(
  "$ROOT/wan22_submission_ti2v_121f_s10.zip"
  "$ROOT/hybrid_wan_region_fixed_flat.zip"
  "$ROOT/hybrid_wan_region_wts_flat.zip"
  "$ROOT/region_norm_medium_submission_flat.zip"
  "$ROOT/hybrid_cnn65_cosmos_bg_s31_a25_wts_guard015_candidate"
  "$ROOT/hybrid_bdd_refiner_v2_wts_guard015_candidate"

  "$WAN/bdd_refiner_router_v2_test"
  "$WAN/bdd_refiner_router_v2_test_quick_sheets"
  "$WAN/bdd_supervised_router_gate24"
  "$WAN/bdd_distribution_rerank_gate24"
  "$WAN/cnn_ltxmix65_cosmos_bg_s31_a25_bdd_test"
  "$WAN/region_norm_routed_wts_test"
  "$WAN/wts_overhead_long24_vehicle_medium_test"

  "$WAN/matched_cnn_ltxmix65_train16"
  "$WAN/temporal_matched_cnn_ltxmix65_train16_gate24"
  "$WAN/temporal_matched_cnn_ltxmix65_train16_gate24_compare_sheets"
  "$WAN/highres_temporal_detail_matched16_gate24"
  "$WAN/highres_temporal_detail_matched16_gate24_compare_sheets"
  "$WAN/bdd_train100_region_motion_base"
  "$WAN/frames_bdd100_materialized"

  "$WAN/ltx_sampler_stg_smoke4"
  "$WAN/ltx_sampler_stg_smoke4_deterministic_decode_sheets"
  "$WAN/ltx_sampler_stg_smoke4_stg_attention_s020_sheets"
  "$WAN/ltx_sampler_stg_smoke4_stg_residual_s020_sheets"
)

# These prefixes are formally closed model/refiner directions. Expand only
# direct children of the experiment root, then validate every resolved path.
while IFS= read -r -d '' path; do targets+=("$path"); done < <(
  find "$WAN" -mindepth 1 -maxdepth 1 \
    \( -name 'cosmos3*' \
       -o -name 'skyreels*' \
       -o -name 'framepack*' \
       -o -name 'motion_locked_texture*' \
       -o -name 'metric_*adapter*' \
       -o -name 'vace_cnn_refiner*' \
       -o -name 'wan22_ti2v_bdd_smoke4*' \
       -o -name 'temporal_latent_feature_train100*' \
       -o -name 'temporal_latent_refiner_smoke16_heldout8*' \) \
    -print0
)

declare -A seen=()
safe_targets=()
total_kib=0
for path in "${targets[@]}"; do
  [[ -e "$path" ]] || continue
  resolved=$(realpath -m -- "$path")
  case "$resolved" in
    "$ROOT"/*) ;;
    *) echo "Refusing path outside $ROOT: $resolved" >&2; exit 2 ;;
  esac
  if [[ -n "${seen[$resolved]:-}" ]]; then continue; fi
  seen[$resolved]=1
  safe_targets+=("$resolved")
  kib=$(du -sk -- "$resolved" | awk '{print $1}')
  total_kib=$((total_kib + kib))
  printf 'delete %8.2f GiB  %s\n' "$(awk -v k="$kib" 'BEGIN {print k/1048576}')" "$resolved"
done

printf 'planned_total=%.2f GiB paths=%d\n' "$(awk -v k="$total_kib" 'BEGIN {print k/1048576}')" "${#safe_targets[@]}"

for path in "${safe_targets[@]}"; do
  resolved=$(realpath -m -- "$path")
  case "$resolved" in
    "$ROOT"/*) rm -rf --one-file-system -- "$resolved" ;;
    *) echo "Refusing changed path outside $ROOT: $resolved" >&2; exit 3 ;;
  esac
done

echo "deleted_paths=${#safe_targets[@]}"
df -h /workspace

echo "Preserved critical folders:"
printf '  %s\n' \
  "$ROOT/hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate" \
  "$ROOT/hybrid_temporal_latent_feature_wts_overhead_object_layer_candidate" \
  "$WAN/cnn_ltxmix65_bdd_test" \
  "$WAN/temporal_latent_feature_bdd_test" \
  "$WAN/wts_overhead_object_layer_vehicle_medium_test" \
  "$WAN/cosmos25_diffusers_env" \
  "$WAN/cosmos25_hf_cache"
