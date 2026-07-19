# HANDOFF 2026-07-07 - AI City Track 5 temporal ghosting / 75+ big-jump search

## TL;DR for the next chatbox

Read this file first, then read the two older handoffs:

1. `/workspace/wan_ticft/HANDOFF_2026_06_29_multiframe.md`
2. `/workspace/wan_ticft/HANDOFF_2026_07_04_bigjump.md`
3. `/workspace/wan_ticft/HANDOFF_2026_07_07_temporal_ghosting_bigjump.md`

Current public score is still around 69.4, not 75+. The main blocker is not PSNR/SSIM. It is perceptual/video realism: temporal ghosting, afterimage, double exposure, road/lane smearing, and stale object trails.

Best known team submissions:

- `mf_h8_actorflow`: General score `69.4176`, metrics pasted by user: `18.6032 / 0.6201 / 0.3301 / 0.9398 / 46.6988 / 31.4066`
- `ltxhf_t55`: Public score `69.3815`, metrics pasted by user: `18.6056 / 0.6194 / 0.3303 / 0.9400 / 46.8985 / 31.1503`
- Older: `actorflow objv1 = 69.3579`, `old mf_h8 = 69.3427`
- Top public pasted by user: `73.1661`, with much better FID/FVD/LPIPS/perceptual side. Target is 75+ for safe win.

Most important diagnosis: the visible artifact is generated geometry error, not ordinary blur/noise. Moving vehicles/pedestrians/lane markings are blended across time. The model does not pick one future state; it averages old and new states, creating semi-transparent duplicates. Later, stale positions become dark road trails. Postprocess/inpainting can slightly change color/sharpness, but cannot reliably remove a wrong temporal geometry structure that is already baked into the frames.

## Environment / access

Local Windows workspace:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n`

Remote workspace:

- `/workspace`
- Main repo/scripts: `/workspace/wan_ticft`
- Experiments: `/workspace/experiments`

SSH pattern from Windows PowerShell:

```powershell
$ask='C:\Users\Windows\Documents\Codex\2026-06-29\b-n-ti-p-qu-n\work\sshfs_askpass.cmd'
$env:SSH_ASKPASS=$ask
$env:SSH_ASKPASS_REQUIRE='force'
$env:DISPLAY='1'
ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no khangle-dev-vguai '<bash command>'
```

Use `/opt/conda/envs/wan22/bin/python` for most Python/OpenCV/torch scripts. For LTX, existing wrapper calls `/workspace/experiments/wan_ticft/ltxv_env/bin/python`.

Disk cleanup was done. At the time of this handoff, `/workspace` had about `877G` free.

## User-visible artifact to solve

Representative sample: `video3806`.

The user repeatedly inspected videos and screenshots and identified:

- In early frames around `f5-f8`, cars and lane/curb markings show double exposure.
- The black/red cars appear simultaneously at old and new positions.
- Lane markings, zebra crossing, curb, and road edges smear into parallel faded copies.
- Later, around `f37`, the old object positions become dark trails on the road.
- The issue is present in the raw/best LTX/LTXHF outputs before any ProPainter/deghost/inpaint.
- The user corrected one earlier assumption: the cars that looked "static" are actually supposed to move too. The problem is whole-scene temporal averaging, not one stationary-object mask bug.

Names for this artifact:

- temporal ghosting
- afterimage artifact
- double exposure artifact
- motion-induced ghosting
- object persistence artifact
- stale-object trail
- occlusion/disocclusion failure
- motion uncertainty averaging
- dark trail / road smear

## What NOT to waste time rerunning

These branches were already explored or explicitly closed in the previous handoff or during this session:

- StableVSR
- pure/chunked LTX as a final answer
- low-frequency/detail sweeps
- adversarial residual adapter
- temporal latent feature
- LoRA scaffold resume
- router param sweep
- simple restoration/sharpening/deblur
- simple inpainting/deghost filters
- ProPainter alone
- SA2VA/SAM2/MaskRCNN object-mask inpainting as a general fix
- object-only patching of the black car
- static mask assumption for `video3806`

They can be referenced for learning, but do not spend another day rerunning them without a genuinely new control/training idea.

## Research read during this phase

### Memory-V2V paper from user PDF `2601.16296v2.pdf`

Useful idea:

- Retrieval/memory videos can provide structured motion priors.
- They can reduce motion uncertainty if used as conditioning/motion anchor.
- Full method requires model-level training/fine-tuning, not just post-hoc blending.

What was adapted:

- "Memory motion anchor" for `video3806`: retrieve similar motion, affine-align, and use as scaffold.

Result:

- It can reduce some early ghost if used strongly, but causes freezing/warping/incorrect trajectory when used too long.
- As a postprocess/scaffold only, it did not solve the issue.

### Reangle-Video

Repo cloned locally:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\research\Reangle-Video`

Paper/repo idea:

- Use depth/camera/warped videos as constraints.
- Important part is not just warping, but masked diffusion loss on visible/confident regions.
- Do not trust the entire warped scaffold. Train/fine-tune model with visibility masks.

What was adapted:

- "Reangle-lite" masked/temporal scaffolds for LTX: anchor only uncertain/ghost zones or early frames, then fade.

Result:

- Reduces some gross ghost in still frames but causes early "khựng" / held motion, then sudden recovery/jump.
- LTX inherits scaffold freeze instead of learning a clean dynamic state.
- This strongly suggests the real transferable idea is masked/visibility-aware training, not scaffold injection alone.

## Experiments and outcomes

### 1. Baseline/best candidates

Best current candidate path:

- `/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate`

Other useful sources:

- `/workspace/experiments/wan_ticft/ltx_detail_donor_bdd24_gate/ltxhf_t55/videos`
- `/workspace/experiments/wan_ticft/basemode_gate/mf_a75_h4_video/videos`
- `/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/videos`
- `/workspace/experiments/wan_ticft/ltx2b_chunked_traffic_bdd24_gate/videos`
- `/workspace/experiments/wan_ticft/ltx2b_chunked_dashcam_native_bdd24_gate/videos`

Verdict:

- Best public/local candidate still around 69.4.
- Good geometry/detail in static regions, but temporal ghosting is obvious in moving regions.

### 2. ProPainter / inpainting / deghost

Tried:

- global deghost
- masked deghost
- dark trail lift
- ProPainter on detected masks
- past-object-position masks
- SA2VA/SAM2/MaskRCNN-style masks

Representative output roots:

- `/workspace/experiments/wan_ticft/object_pastmask_propainter_video3806`
- `/workspace/experiments/wan_ticft/static_car_protect_mask_video3806`
- `/workspace/experiments/wan_ticft/video3806_deghost_video_review_20260706`

Verdict:

- Failed as a main solution.
- Reason: masks track current objects or remove only visible pieces, but the ghost is not isolated noise. It is semi-transparent wrong geometry spread through vehicles, lanes, curbs, road, and background.
- Inpainting often preserves the ghost, smears it, or changes colors only.

### 3. Restoration prior / temporal deghost refiner

Relevant local scripts:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\train_bdd_deghost_restoration_prior.py`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\train_temporal_deghost_refiner_fast.py`

Representative outputs:

- `/workspace/experiments/wan_ticft/deghost_prior_smoke_v3806_s0.6/video3806`
- `/workspace/experiments/wan_ticft/deghost_prior_smoke_v3806_s1.0/video3806`
- `/workspace/experiments/wan_ticft/deghost_prior_smoke_v3806_s1.4/video3806`
- `/workspace/experiments/wan_ticft/masked_deghost_prior_smoke_v3806_s14/video3806`

Verdict:

- Failed for 75+.
- It changes texture/contrast slightly, but does not remove generated temporal geometry.

### 4. VACE / other Wan-style attempts

Verdict:

- Failed visually. VACE changed too much of the scene and did not preserve scaffold/geometry enough.
- User explicitly said "vace tệ quá, nó thay đổi hết mọi thứ".

### 5. Memory motion anchor

Scripts:

- Local: `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\memory_motion_anchor_smoke.py`
- Remote: `/workspace/wan_ticft/memory_motion_anchor_smoke.py`

Important outputs:

- `/workspace/experiments/wan_ticft/memory_motion_anchor_video3806_diverse_clamp`
- `/workspace/experiments/wan_ticft/memory_motion_anchor_video3806_long59`
- `/workspace/experiments/wan_ticft/memory_scaffold_ltx_video3806_s085/ltx_scaffold_locked_full65_notile`

Local comparison videos:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\memory_motion_anchor_video3806_diverse_clamp_compare_20260707`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\memory_motion_anchor_video3806_long59_compare_20260707`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\memory_scaffold_ltx_full65_video3806_compare_20260707\video3806_candidate_compare.mp4`

Verdict:

- Interesting but not good enough.
- Strong anchor can remove some ghost, but freezes/warps objects.
- LTX scaffold-locked output inherits the scaffold's early hold/stutter.

### 6. Reangle-lite masked/temporal scaffolds

Remote root:

- `/workspace/experiments/wan_ticft/reangle_masked_scaffold_video3806`

Outputs:

- `/workspace/experiments/wan_ticft/reangle_masked_scaffold_video3806/ltx_masked_s085_notile`
- `/workspace/experiments/wan_ticft/reangle_masked_scaffold_video3806/ltx_temporal_s085_notile`
- `/workspace/experiments/wan_ticft/reangle_masked_scaffold_video3806/ltx_temporal_strength_sweep_notile/scaf_s055`
- `/workspace/experiments/wan_ticft/reangle_masked_scaffold_video3806/ltx_temporal_strength_sweep_notile/scaf_s070`

Local compare:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\reangle_scaffold_all_video3806_compare_20260707\video3806_candidate_compare.mp4`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\reangle_temporal_strength_compare_video3806_20260707\video3806_candidate_compare.mp4`
- Early audit strips:
  - `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\early_motion_audit_video3806_20260707\grid_f00_f07.jpg`
  - `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\early_motion_audit_video3806_20260707\grid_f08_f15.jpg`

Verdict:

- Failed as post-hoc scaffold.
- It suppresses some ghost in still frame inspection, but video reveals early hold/stutter and object recovery artifacts.

### 7. Flow-propagated clean anchor

Remote root:

- `/workspace/experiments/wan_ticft/flowprop_clean_anchor_video3806`

Local audit:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\flowprop_scaffold_audit_video3806_20260707`

Verdict:

- Failed. Optical flow in ghosted object regions is unreliable because the base itself already contains blended states.

### 8. Temporal unmix deghost

Remote root:

- `/workspace/experiments/wan_ticft/temporal_unmix_deghost_video3806`

Variants:

- `beta045`
- `beta070`
- `beta095`

Local compare:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\temporal_unmix_deghost_video3806_compare_20260707\video3806_candidate_compare.mp4`

Verdict:

- Motion is less frozen than anchor methods, but ghost remains and artifacts appear. Not a 75+ route.

### 9. Object trajectory / object halo attempts

Scripts:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\object_traj_clean_car_video3806.py`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\object_halo_deghost_video3806.py`

Remote roots:

- `/workspace/experiments/wan_ticft/object_traj_clean_car_video3806`
- `/workspace/experiments/wan_ticft/object_halo_deghost_video3806`

LTX output:

- `/workspace/experiments/wan_ticft/object_traj_clean_car_video3806/ltx_object_traj_s070_notile`

Local compare:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\object_traj_clean_car_video3806_compare_20260707`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\object_traj_ltx_video3806_compare_20260707`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\object_halo_deghost_video3806_compare_20260707\video3806_candidate_compare.mp4`

Verdict:

- Failed. Object can move, but deforms, then recovers. Halo version preserves object body but leaves ghost. This is not enough because the artifact spans the whole scene, not only one car.

### 10. Chunk as motion source

Observation:

- Raw chunk does not show the same strong afterimage/double exposure.
- But raw chunk looks fake/glossy, with weak far details, changed geometry, bad distant regions, and sometimes extra/shifted content.
- User noted: chunk is the only one not dragging images, but it does not look real.

Relevant raw chunk:

- `/workspace/experiments/wan_ticft/ltx2b_chunked_traffic_bdd_test_smoke3806/video3806`

#### Chunk motion scaffold -> LTX locked

Remote root:

- `/workspace/experiments/wan_ticft/chunk_motion_scaffold_ltx_video3806`

Outputs:

- `/workspace/experiments/wan_ticft/chunk_motion_scaffold_ltx_video3806/video3806_chunk_motion_scaffold65.mp4`
- `/workspace/experiments/wan_ticft/chunk_motion_scaffold_ltx_video3806/ltx_chunk_motion_sweep_notile/scaf_s055/videos/test_bdd_pc_5k_bdd_video3806_phase2.mp4`
- `/workspace/experiments/wan_ticft/chunk_motion_scaffold_ltx_video3806/ltx_chunk_motion_sweep_notile/scaf_s070/videos/test_bdd_pc_5k_bdd_video3806_phase2.mp4`

Local compare:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_motion_scaffold_ltx_video3806_compare_20260707\video3806_candidate_compare.mp4`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_motion_scaffold_ltx_video3806_compare_20260707\video3806_candidate_keyframes.jpg`

Verdict:

- Partially successful at reducing double exposure.
- But it follows chunk geometry/look too much. It becomes synthetic/fake and can drift from the original scene.
- Not a final submission route as-is.

#### Chunk flow projector

Idea:

- Use chunk only for motion/flow, keep base texture/detail.
- Warp previous corrected base frame according to chunk motion and blend in motion regions.

Remote root:

- `/workspace/experiments/wan_ticft/chunk_flow_projector_video3806`

Local compare:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_flow_projector_video3806_compare_20260707\video3806_candidate_compare.mp4`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_flow_projector_video3806_compare_20260707\video3806_candidate_keyframes.jpg`

Verdict:

- Failed. The base ghost remains because flow/blend cannot cleanly separate real object state from ghosted state.
- Stronger blend makes image softer/less real.

## Current best result

For public score, the best known remains:

- `mf_h8_actorflow` / `ltxhf_t55`, around `69.4`.

For visual diagnosis, the most informative compare is:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_motion_scaffold_ltx_video3806_compare_20260707\video3806_candidate_compare.mp4`

This shows the core tradeoff:

- Base/LTX: realistic texture but temporal ghosting/double exposure.
- Raw chunk: cleaner motion/no ghost but synthetic, poor distant detail, geometry/look drift.
- LTX-from-chunk: less ghost but too much chunk geometry/look, not robust enough.

## Current working hypothesis

The 75+ route is not output filtering. The route must prevent temporal averaging inside generation.

Most plausible high-impact direction:

1. Use raw chunk, memory retrieval, or flow/depth/DINO-style correspondence as a motion teacher.
2. Keep the high-scoring/base geometry and history frames as appearance/scaffold.
3. Fine-tune a controllable video generator or lightweight adapter with a masked temporal/visibility loss:
   - penalize ghost/afterimage regions;
   - preserve visible/confident base geometry;
   - use chunk/memory only as motion target where base is uncertain;
   - avoid full-frame chunk supervision because chunk is visually fake.
4. This is closer to the useful parts of Reangle/Memory-V2V than current post-hoc experiments.

If there is only time for one serious next branch, do this:

- Build a training set from BDD train/val where future GT exists.
- Create pseudo "bad" inputs by running/using current LTXHF/base predictions or synthetic temporal blending.
- Train a small temporal refiner or LoRA/control adapter to map `(history frames, base prediction, chunk/memory motion teacher, visibility/uncertainty mask)` to GT future.
- Loss should be perceptual/LPIPS + temporal consistency + masked L1/Charbonnier, with emphasis on moving-object and disocclusion regions.
- Do not supervise the whole frame from chunk. Only use chunk/memory as motion/confidence guide.

This is hard but is the only direction left that matches the real failure.

## Suggested next experiment design

### Option A: masked temporal refiner trained on real BDD GT

Inputs per sample:

- history frames
- current best base prediction
- chunk prediction or memory/retrieval motion
- motion/uncertainty masks from frame difference, RAFT/Farneback flow inconsistency, detector/segmentation, or temporal variance

Target:

- real GT future frames from train split

Model:

- small U-Net / recurrent video refiner / lightweight temporal transformer
- keep resolution and channels manageable; start with 256/384 crop or 512x896 if GPU allows

Loss:

- masked L1/Charbonnier on motion/ghost regions
- LPIPS/perceptual if available
- temporal warping consistency
- background preservation loss outside mask

Goal:

- Learn to remove generated temporal averaging while preserving base appearance.

Why this is different from failed restoration prior:

- It must include motion teacher/uncertainty masks and real GT supervision, not just generic deblur/restoration.

### Option B: controlled generation fine-tune / LoRA

Train LTX/CogVideo/Wan-style adapter if feasible:

- condition on history + base scaffold + masked motion teacher
- use visibility/confidence masks like Reangle
- avoid full-scene redesign

This is theoretically better but may be too heavy for 3-5 days unless existing training scaffolds already work.

## Commands/patterns that worked

LTX scaffold locked pattern:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace nohup /opt/conda/envs/wan22/bin/python /workspace/wan_ticft/evaluation/run_ltxv_scaffold_locked_gate.py \
  --manifest /workspace/experiments/wan_ticft/memory_scaffold_ltx_video3806_s085/manifest_video3806.jsonl \
  --scaffold_report <report.csv> \
  --out_dir <OUT> \
  --limit 1 --sample_ids test_bdd_pc_5k_bdd_video3806_phase2 \
  --strengths 0.55 0.70 \
  --future_condition_frames 65 --max_history_frames 17 \
  --height 512 --width 896 \
  --image_cond_noise_scale 0.025 \
  --vae_tiling 0 --skip_bad > <LOG> 2>&1 &
```

Important: `--vae_tiling 0` was required. VAE tiling caused failures with conditioning video temporal latent depth 9.

Compare-video builder:

- Remote: `/workspace/wan_ticft/build_candidate_compare_video.py`
- Local copy: `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\work\build_candidate_compare_video.py`

Example:

```bash
/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/build_candidate_compare_video.py \
  --sample video3806 --out_dir /workspace/experiments/wan_ticft/some_compare --fps 10 \
  base=/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate \
  raw_chunk=/workspace/experiments/wan_ticft/ltx2b_chunked_traffic_bdd_test_smoke3806 \
  candidate=/workspace/experiments/wan_ticft/some_candidate_frame_root
```

## Local videos to show user quickly

Chunk/LTX tradeoff:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_motion_scaffold_ltx_video3806_compare_20260707\video3806_candidate_compare.mp4`

Chunk flow projector failure:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_flow_projector_video3806_compare_20260707\video3806_candidate_compare.mp4`

Reangle scaffold:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\reangle_scaffold_all_video3806_compare_20260707\video3806_candidate_compare.mp4`
- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\reangle_temporal_strength_compare_video3806_20260707\video3806_candidate_compare.mp4`

Memory scaffold:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\memory_scaffold_ltx_full65_video3806_compare_20260707\video3806_candidate_compare.mp4`

Object trajectory:

- `C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\object_traj_ltx_video3806_compare_20260707`

## Suggested prompt for a new chatbox

Paste this into the new chat:

```text
Bạn tiếp quản project AI City Track 5 future-frame prediction trên /workspace.

Đọc đầy đủ 3 handoff này trước khi làm gì:
1. /workspace/wan_ticft/HANDOFF_2026_06_29_multiframe.md
2. /workspace/wan_ticft/HANDOFF_2026_07_04_bigjump.md
3. /workspace/wan_ticft/HANDOFF_2026_07_07_temporal_ghosting_bigjump.md

Mục tiêu vẫn là public 75+, deadline 2026-07-11. Current best public/general của team chỉ khoảng 69.4:
- mf_h8_actorflow General 69.4176
- ltxhf_t55 Public 69.3815

Top public user pasted là 73.1661, gap chính nằm ở FID/FVD/LPIPS/perceptual, không phải PSNR/SSIM. Local pixel metrics từng lừa rất nhiều lần; chỉ tin public score + REAL-FID gate + video visual.

Hiện tượng lớn nhất cần giải quyết:
Trong video future prediction, khi xe/người/camera chuyển động, model blend nhiều trạng thái thời gian lại với nhau, gây temporal ghosting / afterimage / double exposure. Ở video3806, f5-f8 xe và lane/curb bị kéo thành nhiều bản mờ; về sau ghost biến thành dark trail trên road. Lỗi đã có trong output gốc LTX/LTXHF, trước khi ProPainter/deghost/inpaint.

Đừng rerun các nhánh đã fail/đóng nếu không có ý tưởng mới thật sự:
StableVSR, pure/chunked LTX final, low-frequency/detail sweeps, adversarial residual adapter, temporal latent feature, LoRA scaffold resume, router param sweep, ProPainter đơn giản, SA2VA/SAM2/MaskRCNN inpaint, object-only black-car patch, VACE, restoration/deblur/sharpen đơn giản.

Quan trọng:
- Raw chunk gần như không bị kéo ảnh nhưng nhìn synthetic/glossy, sai distant detail/geometry.
- Base/LTX nhìn thật hơn nhưng bị temporal ghosting.
- LTX conditioned bằng chunk motion giảm ghost nhưng bị synthetic/geometry drift.
- Flow/projector từ chunk về base vẫn không xóa được ghost.

Các video compare quan trọng trên local:
- C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_motion_scaffold_ltx_video3806_compare_20260707\video3806_candidate_compare.mp4
- C:\Users\Windows\Documents\Codex\2026-07-04\b-n-ti-p-qu-n\outputs\chunk_flow_projector_video3806_compare_20260707\video3806_candidate_compare.mp4

Hướng tiếp theo nên làm:
Không tiếp tục vá output bằng filter. Cần controlled generation/fine-tuning/retrieval that preserves geometry. Ý tưởng mạnh nhất là học từ Reangle/Memory-V2V:
- dùng chunk/memory/retrieval/flow như motion teacher;
- dùng base/high-score prediction như appearance/geometry scaffold;
- train masked temporal refiner hoặc LoRA/control adapter trên BDD train có GT future;
- loss tập trung vào moving-object/disocclusion/ghost regions, không supervise toàn frame từ chunk vì chunk giả;
- preserve background/static geometry ngoài mask.

Việc đầu tiên:
1. Mở handoff 2026-07-07 và xem lại các video compare bằng mắt.
2. Kiểm tra BDD train/GT có sẵn để build dataset cho masked temporal refiner.
3. Thiết kế experiment nhỏ trên video3806 + vài sample trước, nhưng hướng phải có khả năng scale toàn test.
4. Nếu thử gì, luôn dựng video compare trước khi kết luận. User không muốn nhìn frame tĩnh nữa.
```

## Final recommendation

Be honest with the user: postprocess has essentially hit a wall. The only credible 75+ attempt left is a training/control-based method that prevents temporal averaging during generation or learns to undo it from real GT with motion-aware masks. Chunk is valuable as a motion teacher, not as final visual output.
