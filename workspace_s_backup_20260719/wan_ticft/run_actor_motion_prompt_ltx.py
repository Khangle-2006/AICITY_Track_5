from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROMPT = (
    "Continue this exact real dashcam video. Static background such as buildings, curb, sidewalk, "
    "lane markings, road plane and sky must keep the same geometry and only follow camera ego-motion. "
    "Visible vehicles are dynamic actors, not background: cars in the same lane and neighboring lanes "
    "keep moving forward smoothly with traffic, changing position and scale consistently. Preserve object "
    "identity and realistic road scale. Use the actor motion scaffold only as a trajectory hint, not as "
    "synthetic texture. No double exposure, no ghost trails, no frozen cars, no new objects, no scene redesign."
)

NEGATIVE = (
    "new scene, hallucinated vehicle, object replacement, warped road, lane drift, double exposure, "
    "ghosting, smear, synthetic texture, frozen cars, flicker, low quality"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.55, 0.70])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--ltx_repo", type=Path, default=Path("/workspace/third_party/LTX-Video"))
    parser.add_argument("--ltx_python", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_env/bin/python"))
    parser.add_argument("--pipeline_config", type=Path, default=Path("/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml"))
    args = parser.parse_args()

    for strength in args.strengths:
        label = f"scaf_s{int(round(strength * 100)):03d}"
        raw = args.out_dir / "_raw" / label / "test_bdd_pc_5k_bdd_video3806_phase2"
        if raw.exists():
            shutil.rmtree(raw)
        raw.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.ltx_python),
            str(args.ltx_repo / "inference.py"),
            "--prompt",
            PROMPT,
            "--negative_prompt",
            NEGATIVE,
            "--image_cond_noise_scale",
            "0.0250",
            "--height",
            "512",
            "--width",
            "896",
            "--num_frames",
            "81",
            "--frame_rate",
            "30",
            "--seed",
            str(args.seed),
            "--offload_to_cpu",
            "true",
            "--pipeline_config",
            str(args.pipeline_config),
            "--output_path",
            str(raw),
            "--conditioning_media_paths",
            str(args.history),
            str(args.scaffold),
            "--conditioning_start_frames",
            "0",
            "16",
            "--conditioning_strengths",
            "1.0",
            f"{strength:.4f}",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.ltx_repo)
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["LTX_VAE_TILING"] = "0"
        subprocess.run(command, check=True, env=env)
        print(raw, flush=True)


if __name__ == "__main__":
    main()
