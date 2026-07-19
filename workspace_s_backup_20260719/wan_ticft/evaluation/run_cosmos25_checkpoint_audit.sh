#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/cosmos-predict2.5
PYTHON=/opt/conda/envs/wan22/bin/python

echo "=== Cosmos-Predict2.5 2B checkpoint and deployment audit ==="
echo "Commit: $(git -C "$REPO" rev-parse --short HEAD)"
echo
echo "Host:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
ldd --version | head -1
df -h /workspace
echo

echo "Official inference contract:"
grep -nE \
  'num_output_frames|num_steps|device_map|video2world|offload_diffusion_model|offload_text_encoder|offload_tokenizer' \
  "$REPO/scripts/diffusers_inference.py" "$REPO/cosmos_predict2/config.py" \
  | head -n 120
echo

echo "Hugging Face checkpoint metadata:"
"$PYTHON" - <<'PY'
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

api = HfApi()
repos = [
    ("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"),
    ("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/pre-trained"),
    ("nvidia/Cosmos-Reason1-7B", None),
    ("nvidia/Cosmos-Guardrail1", None),
]
for repo_id, revision in repos:
    revision_label = revision or "main"
    print(f"\n{repo_id} @ {revision_label}")
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except HfHubHTTPError as exc:
        print(f"ACCESS_ERROR {exc}")
        continue
    siblings = list(info.siblings or [])
    total = sum(int(item.size or 0) for item in siblings)
    print(f"files={len(siblings)} total_gib={total / 1024**3:.2f}")
    components = {}
    for item in siblings:
        component = item.rfilename.split("/", 1)[0]
        components[component] = components.get(component, 0) + int(item.size or 0)
    print("components:")
    for component, size in sorted(components.items(), key=lambda pair: pair[1], reverse=True):
        print(f"  {size / 1024**3:7.2f} GiB  {component}")
    print("largest files:")
    for item in sorted(siblings, key=lambda x: int(x.size or 0), reverse=True)[:12]:
        print(f"  {(item.size or 0) / 1024**3:7.2f} GiB  {item.rfilename}")
PY

echo
echo "Audit rule:"
echo "  - Continue only if the 2B post-trained Diffusers revision is accessible,"
echo "    total download/storage is practical, and no required gated dependency"
echo "    blocks inference."
echo "  - If pass, install the isolated Diffusers script dependencies and run one"
echo "    17-frame Video2World smoke with device_map/offload. Do not start at 93 frames."
echo "  - Stop on access failure, >48GB single-GPU incompatibility, or major setup repair."
