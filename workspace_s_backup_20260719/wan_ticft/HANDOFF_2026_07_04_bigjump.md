# AI City Track 5 Handoff - 2026-07-04 Big-Jump Status

## TL;DR
Goal is still **75+ public**, deadline **2026-07-11**. Do **not** spend time on +0.1/+0.2 param sweeps. The user wants a real method jump and paper-worthy visuals.

Known best public for our team:
- `hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate`: **69.3579** public.
- Old `mf_h8`: **69.3427** public.

External top1 row user pasted:
- Top1: `73.1661 18.9483 0.6092 0.2727 0.9359 31.1354 22.4018`
- Ours reference: `69.3325 18.6129 0.6202 0.3313 0.9397 47.1404 31.2713`
Interpretation: our PSNR/SSIM are not the main problem; the gap is mostly perceptual/temporal/distribution, especially FID/FVD/LPIPS-like metrics. User is correct that tiny parameter tuning will not reach 75.

Hard lesson: local metrics have lied many times. Trust only:
1. public score, and
2. REAL-FID gate (`pytorch-fid` vs real GT frames) because it tracked public ranking better than pixel/proxy metrics.

## Current Highest / Pending Candidates

### Highest confirmed public
`hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate`
- Public: **69.3579**.
- Only tiny gain over `mf_h8`; not enough for 75.
- This showed that actor/object replacement has a small real signal but not enough.

### Most promising pending/public-wait candidate
`hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate`
- Folder: `/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate`
- Zip: `/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate.zip`
- Expected file count: **5194 PNG**.
- Size: about **5.9 GB**.
- Method: keep MF h8 geometry/motion, inject high-frequency detail from LTX traffic donor through edge/detail mask. It visibly reduces afterimage/blur but does not fully solve it.
- Gate on BDD24:
  - `ltxhf_t55 REAL-FID = 62.437`
  - `mf_h4 REAL-FID = 68.909`
- User visually said it is much less blurry/afterimage than older folders, so this is currently the safest pending candidate to submit/wait for.
- Caveat: this is still not likely a 75+ route by itself.

Robocopy command if needed:
```bat
robocopy "S:\experiments\hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate" "G:\AICITY_Submissions\hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate" /E /Z /MT:8 /R:10 /W:5 /NP /TEE /LOG+:"G:\AICITY_Submissions\robocopy_ltxhf_t55.log"
```
Check count:
```bat
dir /S /B "G:\AICITY_Submissions\hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate\*.png" | find /C ".png"
```
Expected: `5194`.

## Important Scores / Gates Already Run

### Public scores
- `mf_h8`: **69.3427** public.
- `hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate`: **69.3579** public.
- `ltxhf_t55`: pending/unknown public as of this handoff.

### LTX high-frequency donor variants
BDD24 REAL-FID:
- `ltxhf_t55`: **62.437**
- `ltxhf_t75`: **62.307**
- `ltxhf_t95`: **62.359**
- `real_ltx_t55r25`: **62.270**
These are tiny differences. Do not present t75/t95 as big improvements. `t55` is the stable submission candidate already built.

Low-frequency/strong real-detail variants were bad:
- `ltxlf_b12/b20/real_ltx_lf12`: quick REAL-FID about **167**, closed.
- `real_ltx_r40/r60/realonly_r50/real_ltx_t75r35`: quick REAL-FID about **166**, closed.
Conclusion: adding detail is useful only as high-frequency donor. Low-frequency/color/real-pixel injection destroys distribution.

### LTX chunked / raw video-prior branches
BDD24 chunked gate:
- `ltx2b_chunked`: pixel good but synthetic.
- REAL-FID:
  - `depth_a75`: **70.725**
  - `traffic_detail`: **92.765**
  - `dashcam_native`: **107.343**
Conclusion: pure/chunked LTX is too synthetic. However LTX high-frequency residual as donor helps.

### StableVSR on `ltxhf_t55`
BDD4 full output/proxy:
- `ltxhf_t55`: FID proxy **10.5742**, FVD proxy **26.8917**
- `stablevsr`: FID proxy **10.5631**, FVD proxy **27.7151**
Temporal residual:
- `tmblend_a35`: FID proxy **10.6678**, FVD proxy **28.2884**
Conclusion: fail. StableVSR does not give the jump; closes this path.

### LTX scaffold-LoRA / depth-scaffold LoRA
Folder: `/workspace/experiments/wan_ticft/ltx2b_depth_scaffold_lora_bdd_train48_s200`
Validation score:
- `scaffold`: FID proxy **9.4618**, FVD proxy **17.9846**
- `step0`: FID proxy **19.4067**, FVD proxy **45.4499**
- `step100`: FID proxy **24.4240**, FVD proxy **50.3172**
- `step200`: FID proxy **24.7332**, FVD proxy **46.7189**
Conclusion: fails before training even helps. LTX validation/inference does not preserve reference scaffold; resume training is not useful unless the conditioning mechanism is fixed.

### Adversarial residual adapter on `ltxhf_t55`
Folder: `/workspace/experiments/wan_ticft/adversarial_residual_ltxhf_t55_train16_eval8`
Held-out eval:
- REAL-FID `ltxhf_t55`: **87.005**
- REAL-FID `adversarial_adapter`: **160.883**
- Proxy on 2 complete clips:
  - `ltxhf_t55`: FIDp **6.5571**, FVDp **25.7554**
  - `adversarial_adapter`: FIDp **7.4422**, FVDp **28.2898**
Conclusion: closed. GAN/residual adapter damages distribution badly.

### Temporal latent feature/refiner
Folder: `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24`
- Pixel improves a lot locally.
- Proxy worsens:
  - `cnn_ltxmix65`: FIDp **4.9677**, FVDp **22.2935**
  - `temporal_feature`: FIDp **5.5718**, FVDp **23.1755**
Conclusion: pixel-improving learned refiners are still local-metric traps.

### Flow/highres detail refiners
Old flow/highres refiners improved pixel but worsened proxy.
Fresh `flow_frozen_ltxhf` on `ltxhf_t55` heldout:
- `ltxhf_t55 REAL-FID = 69.376`
- `flow_frozen_ltxhf REAL-FID = 77.244`
Conclusion: closed.

### VACE / Wan / FramePack / Vista raw
- VACE MF-control BDD4 REAL-FID bad: `mf_h4 119.619`, `vace_s50 158.581`, `vace_s30 222.282`.
- Direct BDD refiner v2: `refiner_v2 REAL-FID 120.161`, `mf_h4 68.909`, `depth_a75 70.725`; closed.
- FramePack smoke: lower PSNR/SSIM and geometry drift; closed.
- Wan2.2 smoke: worse than cnn_ltxmix65.
- Vista raw IMG: worse FID/FVD; closed unless a stronger full-history/state conditioning path is built.

### Router/oracle routes
Candidate router/oracle did not open a 75+ path.
- `bdd_candidate_router_train45_eval24/eval24_proxy.csv`:
  - `depth_a75`: FIDp **6.9180**, FVDp **19.2549**
  - `actor_small`: FIDp **5.9788**, FVDp **25.1205**
  - `router`: FIDp **9.1962**, FVDp **23.3101**
  - `oracle`: FIDp **6.9161**, FVDp **17.6891**
Conclusion: oracle improves FVD a bit, real router fails; not enough for 75.

## What To Look At Visually
The user noticed that almost all MF-like folders have faint afterimage/blur trails behind moving objects. The `ltxhf_t55` folder reduces this but does not eliminate it. New chatbox should inspect images, not only metrics.

### Highest known/pending visual folders to inspect
1. Current best pending candidate:
   - `/workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate`
   - BDD test component: `/workspace/experiments/wan_ticft/depth_multiframe_mf_h8_ltxhf_t55_bdd_test`
   - BDD gate component: `/workspace/experiments/wan_ticft/ltx_detail_donor_bdd24_gate/ltxhf_t55/videos`
2. Current highest public baseline-like candidate:
   - `/workspace/experiments/hybrid_multiframe_mf_h8_actorflow_objv1_wts_candidate` if still present.
3. Old SOTA baseline to compare:
   - `/workspace/experiments/wan_ticft/basemode_gate/mf_a75_h4_video/videos`
   - `/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/videos`
4. Chunked LTX examples for synthetic/no-afterimage comparison:
   - `/workspace/experiments/wan_ticft/ltx2b_chunked_traffic_bdd24_gate/videos`
   - `/workspace/experiments/wan_ticft/ltx2b_chunked_dashcam_native_bdd24_gate/videos`

### What to inspect
- Moving cars/pedestrians: is there ghosting/afterimage behind them?
- Road/lanes/sign poles: are they smeared or duplicated?
- Does the candidate look real or synthetic?
- Does it preserve geometry or hallucinate new road/vehicles?
- Night scenes: check light trails and windshield reflections.
- WTS overhead/vehicle views: make sure no corrupted PNGs, no static repeats where motion should exist.

Important: Windows Photos may not open FFV1 `.avi`. Use Python/OpenCV or create contact sheets. Prefer reading video frames and writing PNG contact sheets.

## Important Files / Scripts To Read First In New Chatbox
Read these in order:
1. `/workspace/wan_ticft/HANDOFF_2026_06_29_multiframe.md`
2. `/workspace/wan_ticft/HANDOFF_2026_07_04_bigjump.md` (this file)
3. Key method scripts:
   - `/workspace/wan_ticft/evaluation/ltx_detail_donor.py`
   - `/workspace/wan_ticft/evaluation/ltx_detail_donor_test.py`
   - `/workspace/wan_ticft/evaluation/detail_donor_plus.py`
   - `/workspace/wan_ticft/evaluation/detail_donor_real_sweep.py`
   - `/workspace/wan_ticft/evaluation/compute_real_fid.py`
   - `/workspace/wan_ticft/evaluation/audit_perceptual_video_distribution.py`
   - `/workspace/wan_ticft/evaluation/evaluate_video_reports.py`
4. Closed but informative scripts:
   - `/workspace/wan_ticft/evaluation/run_stablevsr_depth_gate.py`
   - `/workspace/wan_ticft/evaluation/materialize_detailer_temporal_residual_blends.py`
   - `/workspace/wan_ticft/evaluation/train_adversarial_residual_adapter.py`
   - `/workspace/wan_ticft/evaluation/train_metric_residual_adapter.py`
   - `/workspace/wan_ticft/evaluation/run_ltx2b_depth_scaffold_lora_bdd_smoke.sh`
   - `/workspace/wan_ticft/evaluation/score_ltx_scaffold_lora_validation.py`

## Important Result Folders / CSVs
- `/workspace/experiments/wan_ticft/ltx_detail_donor_bdd24_gate/eval_pixel.csv`
- `/workspace/experiments/wan_ticft/detail_donor_plus_bdd24_gate`
- `/workspace/experiments/wan_ticft/realfid_ltx2b_dashcam_native_bdd24`
- `/workspace/experiments/wan_ticft/stablevsr_ltxhf_t55_bdd4_480x270_s10_clip/eval_proxy.csv`
- `/workspace/experiments/wan_ticft/stablevsr_ltxhf_t55_temporal_residual_bdd4/eval_proxy.csv`
- `/workspace/experiments/wan_ticft/ltx2b_depth_scaffold_lora_bdd_train48_s200/validation_score/proxy.csv`
- `/workspace/experiments/wan_ticft/adversarial_residual_ltxhf_t55_train16_eval8_proxy.csv`
- `/workspace/experiments/wan_ticft/realfid_adversarial_residual_ltxhf_t55_eval8`
- `/workspace/experiments/wan_ticft/temporal_latent_feature_gate24_proxy.csv`
- `/workspace/experiments/wan_ticft/bdd_candidate_router_train45_eval24/eval24_proxy.csv`

## Strategic Conclusion
The current best practical candidate is `ltxhf_t55`, but it likely will not hit 75.

Closed paths:
- More alpha/param tuning of detail donor.
- StableVSR/VSR postprocess.
- Pure/chunked LTX.
- Low-frequency LTX/real-detail injection.
- Pixel-trained refiner/adapters.
- Adversarial residual adapter as currently implemented.
- LTX scaffold LoRA unless conditioning is fundamentally fixed.
- Router/oracle as a main 75+ route.

Most likely 75+ route must be **true controlled video generation/fine-tuning that preserves scaffold/geometry** while improving realism/FID/FVD. Existing attempts failed because video-prior models regenerated the scene instead of staying locked to condition.

## Next High-Potential Research Questions
Do not research tiny filters. Research mechanisms that can enforce geometry/identity while using a video prior:
1. How to hard-condition video diffusion on a reference/scaffold video without losing geometry.
   - Search terms: `video diffusion control reference video condition preserving structure`, `ControlNet video diffusion optical flow condition`, `LTX Video reference video conditioning LoRA`, `video-to-video diffusion structure preservation`.
2. Whether any open video model has true video-to-video editing with strong structure preservation and can run locally before deadline.
   - Candidates to investigate: CogVideoX video-to-video/control, Wan/VACE control modes, LTXV img/video-to-video settings, SVD/AnimateDiff with ControlNet/ControlNet-Tile if feasible.
3. Whether training should be done as denoising/refinement around scaffold rather than generation from text/reference.
   - Key idea: add noise to GT/scaffold latents and train to denoise while conditioning on scaffold; inference starts from scaffold/noise, not empty generation.
4. Whether challenge scoring rewards repeating/low-motion for some phases and generated motion for others, but only if public/gate agrees.
5. Whether BDD/WTS domain split requires separate methods: BDD dashcam vs WTS overhead/vehicle may need different priors.

## Commands / Environment Reminders
Remote via Windows PowerShell usually works with:
```powershell
$env:SSH_ASKPASS=(Resolve-Path 'work\sshfs_askpass.cmd').Path
$env:SSH_ASKPASS_REQUIRE='force'
$env:DISPLAY='1'
ssh -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no khangle-dev-vguai "cd /workspace && <cmd>"
```
Conda env:
```bash
/opt/conda/envs/wan22/bin/python
```
Keep CPU capped where scripts support it. Avoid long blind renders; gate small first.

## Prompt For New Chatbox
Copy/paste this into the new chatbox:

```text
B?n ti?p qu?n project AI City Track 5 future-frame prediction tr?n /workspace.

??c tr??c 2 file handoff n?y, ??y ??:
1. /workspace/wan_ticft/HANDOFF_2026_06_29_multiframe.md
2. /workspace/wan_ticft/HANDOFF_2026_07_04_bigjump.md

M?c ti?u: public 75+, deadline 2026-07-11. User kh?ng mu?n t?i ?u +0.1/+0.2 n?a; c?n c? nh?y th?t v? visual paper-worthy.

Tr?ng th?i hi?n t?i:
- Best public ?? bi?t c?a team: actorflow objv1 = 69.3579; old mf_h8 = 69.3427.
- Candidate ?ang ??ng ch? public nh?t: /workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate, zip c?ng t?n trong /workspace/experiments. Gate BDD24: REAL-FID ltxhf_t55 = 62.437 vs mf_h4 = 68.909. N? gi?m afterimage/blur nh?ng ch?a ch?c ?? 75.
- Top1 public user pasted: 73.1661, gap c?a ta ch? y?u l? FID/FVD/LPIPS/perceptual, kh?ng ph?i PSNR/SSIM.

Vi?c ??u ti?n:
1. ??c handoff 2026-07-04 bigjump.
2. Kh?ng ch?y l?i c?c nh?nh ?? ??ng: StableVSR, pure/chunked LTX, low-frequency/detail sweeps, adversarial residual adapter, temporal latent feature, LoRA scaffold resume, router param sweep.
3. M?/soi ?nh v? video c?a c?c folder quan tr?ng:
   - /workspace/experiments/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate
   - /workspace/experiments/wan_ticft/ltx_detail_donor_bdd24_gate/ltxhf_t55/videos
   - /workspace/experiments/wan_ticft/basemode_gate/mf_a75_h4_video/videos
   - /workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/videos
   - /workspace/experiments/wan_ticft/ltx2b_chunked_traffic_bdd24_gate/videos
   - /workspace/experiments/wan_ticft/ltx2b_chunked_dashcam_native_bdd24_gate/videos
4. Khi soi ?nh, t?m ghosting/afterimage, synthetic look, geometry drift, duplicated cars/pedestrians, road/lane smearing, night light trails.
5. ?? xu?t h??ng 75+ th?t: controlled video generation/fine-tuning that preserves scaffold/geometry, kh?ng ph?i postprocess/filter. Research k? c?ch hard-condition video diffusion on scaffold/video-to-video control/denoising around scaffold.

Lu?n nh?: local pixel metrics t?ng l?a nhi?u l?n. Ch? tin public score + REAL-FID gate. M?i nh?nh m?i ph?i BDD small gate -> REAL-FID/proxy -> visual sheets -> m?i scale/build test.
```
