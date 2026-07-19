# AI City Track 5 — HANDOFF 2026-06-29 (multi-frame fusion direction)

## 0. SETUP / MÔI TRƯỜNG (đọc trước)
- Chạy trong **container**: `conda activate wan22`, cwd `/workspace` (= ổ Windows `S:\`). Python `/opt/conda/envs/wan22/bin/python`.
- `/data` chỉ có TRONG container (frame input test). `/workspace` = `S:\` (mount CHẬM, thỉnh thoảng flap — `ls`/`find` hay timeout; ghi file/script nên dùng python, không heredoc dài).
- FID gate cần `pytorch-fid` (đã pip install trong wan22). Làm ảnh montage: dùng Windows `C:\Users\Windows\AppData\Local\Programs\Python\Python311\python.exe` (có cv2) phía ngoài container.
- **CPU cap**: mọi job render đã cap 4 core (`CPU_THREADS=4` + OMP/MKL exports + `cv2.setNumThreads`). ĐỪNG để ăn full core (gây lag server — từng bị 2000%).
- Submit: robocopy folder ra `G:\AICITY_Submissions\`, zip bằng PowerShell, upload. KHÔNG zip trên server.

## 1. MỤC TIÊU
- Vượt best public và tiến tới **75+**. Deadline **2026-07-11**.
- **CỐT LÕI:** local metrics TỪNG LỪA liên tục (proxy FID local ~6 nhưng public 60-76; local PSNR 21 nhưng public-tương-đương 18). CHỈ tin: (a) public score thật, (b) REAL-FID gate (pytorch-fid vs GT thật — ĐÃ chứng minh tái hiện đúng ranking public 3 lần). Mọi "thắng" trên local PHẢI xác nhận bằng public hoặc REAL-FID gate.

## 2. SUBMISSION FORMAT
- 71 folder (44 BDD tên `videoXXXX` + 27 WTS tên `<scene>`), mỗi folder PNG `0.png..(N-1).png`, đúng `target_N` frame, **1280×720**.
- Build: `python -m wan_ticft.evaluation.build_hybrid_submission --wan_dir <BDD_dir> --region_dir <WTS_dir> --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera --out_dir <sub> --report_csv <csv> --clean`.
- Verify: `python -m baseline.submission --gen_dir <sub> --submission_dir <sub> --verify_only` (phải 71 ok, 0 bad).
- Test index: `experiments/index/test_index.csv` (44 bdd + 27 wts).

## 3. PUBLIC SCORES (leaderboard; điểm = avg PSNR/SSIM/LPIPS/CLIP-S/FVD; FID hiển thị riêng)
| Name | Score | PSNR | SSIM | LPIPS | CLIP | FID | FVD |
|---|---|---|---|---|---|---|---|
| a75 (depth-warp + WTS) — best CŨ | 69.2456 | 18.5942 | 0.6222 | 0.3312 | 0.9397 | 47.832 | 31.277 |
| **mf_h8 (MULTIFRAME = BEST HIỆN TẠI)** | **69.3427** | 18.6125 | 0.6204 | 0.3309 | 0.9399 | 47.156 | 31.232 |
| mf_h12 (history sâu) — flat | 69.3325 | 18.6129 | 0.6202 | 0.3313 | 0.9397 | 47.140 | 31.271 |
| overhead_repeat (a75, WTS overhead->repeat) — flat | 69.2656 | 18.5745 | 0.6223 | 0.3312 | 0.9395 | 47.576 | 31.636 |
| lightF (a75 + degradation nhẹ) — wash | 69.2577 | 18.5855 | 0.6196 | 0.3324 | 0.9302 | 46.418 | 31.212 |
| mf_h8_chunk (multiframe + chunk fill) — tệ hơn | 69.1080 | 18.5856 | 0.6155 | 0.3374 | 0.9390 | 47.168 | 31.740 |
| chunk_degr (chunk+depthwarp+degr) | 67.2980 | 18.2713 | 0.6147 | 0.3457 | 0.9167 | 53.643 | 34.081 |
| chunked_dpt (chunk+depthwarp) | 66.2818 | 18.2502 | 0.6153 | 0.3460 | 0.9279 | 60.775 | 34.347 |
| Chunk_ltx2b_res (chunk thuần 1280x720) | 65.0839 | 18.18 | 0.6133 | 0.3516 | 0.9236 | 66.575 | 34.479 |
| Chunk_ltx2b (chunk 704x1216 SAI res) | 62.7370 | 18.19 | 0.6043 | 0.3708 | 0.9144 | 76.353 | 37.517 |

## 4. REAL-FID GATE (val BDD24, pytorch-fid vs GT thật; THẤP = tốt; TÁI HIỆN ĐÚNG RANKING PUBLIC)
- a75_single = **70.725** ; mf_video (multiframe base=ltxv) = **68.909** ; chunk_fixed = 92.6 ; chunk_adaptive 93.4 ; chunk_a00 (chunk thuần) 102.8.
- Degradation sweep trên a75: light(blur0.4 noise3 jpeg50) = **68.2** (thấp nhất); raw 70.7; ép mạnh HẠI (blur-only0.8=74, noise-only6=73.5).
- Tool: `compute_real_fid.py` (`--candidate name=video_dir` lặp lại; optional `--blur/--noise/--jpeg_q/--downscale`). Real GT lấy từ `frames_balanced_val_125/val/<sid>/target/*.png`. Lưu ý: số tuyệt đối (~70) khác public (~48) vì chỉ 216 frame GT thưa — chỉ tin THỨ TỰ.

## 5. HƯỚNG ĐANG THẮNG: MULTI-FRAME FUSION
- File: `materialize_depth_layered_multiframe.py` (val), `materialize_depth_layered_multiframe_test.py` (test). Build: `build_multiframe_wts_candidate.sh` (env: MAX_HISTORY, HISTORY_STRIDE, CHUNK_DIR, BASE_MODE, TAG, CPU_THREADS).
- Cơ chế: warp NHIỀU history frame (mỗi frame: depth riêng + pose `h_i->anchor`, compose với pose `anchor->target`) vào từng target, z-composite kiểu painter (anchor đè trên cùng, history cũ lấp disocclusion bằng PIXEL THẬT). Sửa "kéo/drag" của a75 single-anchor. Helpers import từ `materialize_depth_layered_renderer` (camera_matrix, estimate_pose, zbuffer_splat, render_confidence, smooth_pose_sequence, compose, compose_pose).
- Kết quả: **mf_h8 = 69.34 (BEST, +0.097 vs a75)**. History bão hoà ~8 (h12 flat). Val Pareto: PSNR/SSIM↑ VÀ REAL-FID 70.7->68.9.

## 6. VẤN ĐỀ CHÍNH ĐANG KHẮC PHỤC: GHOSTING (hình-chồng-hình)
- Vật chuyển động (xe, biển, người) hiện thành **2 bản (double)**. Gốc: base `cnn_ltxmix65` = average CỐ ĐỊNH `0.35*region_warp + 0.65*LTX_transition16` -> 2 prediction lệch nhau bị trộn -> double. a75 VÀ multiframe đều build trên base này nên đều ghost (rõ nhất ở frame xa + vật động).
- Input để RE-BLEND (cnn `cnn_ltx_region_selector_bdd_test` + ltx `ltxv_bdd_test_transition16`) đã BỊ XOÁ trước đó. KHÔNG re-blend trực tiếp được.
- **Đang thử fix:** `--base_mode repeat_anchor` = bỏ base ghost; pose VẪN lấy từ cnn base (bg warp tiến CÓ motion) NHƯNG content lấp = ANCHOR THẬT FROZEN (biến `fillbase` trong code) -> foreground 1 bản, không ghost, không synthetic.
  - **BUG lần đầu (đã sửa):** ban đầu set `bases=[anchor]` -> pose(anchor vs anchor)=identity -> VIDEO ĐỨNG YÊN HOÀN TOÀN (= repeat_last). Val PSNR 19.97 / REAL-FID 60.16 khi đó là GIẢ (chữ ký repeat_last tĩnh, user phát hiện video không chuyển động). ĐÃ SỬA: tách pose(từ cnn) khỏi content(anchor) bằng `fillbase`.
  - **TRẠNG THÁI: bản fixed CHƯA re-gate.** PHẢI chạy lại gate (mục 8 bước 1) + KIỂM TRA video repeat_anchor đã có motion lại chưa (frame đầu vs cuối phải khác nhau).

## 7. ĐÃ THỬ & CHẾT (ĐỪNG LẶP LẠI)
- **IC-LoRA mọi biến thể** (camera-style LoRA, IC-LoRA WTS, Refiner-A region->GT, Refiner-B chunk->GT): không hiệu quả, plumbing dữ liệu mong manh (train video BDD decode lệch ~77%). BỎ HẲN.
- **chunk mọi dạng**: standalone 66.28, làm base cho depth-warp 66.28, residual fill 69.11 — đều tệ hơn pure multiframe/a75. Texture synthetic -> FID cao, lại mềm. BỎ.
- **degradation** (blur+noise+jpeg hậu xử lý): chỉ TRADE (FID xuống, PSNR/SSIM/CLIP xuống) -> net ~0 (lightF 69.26 ≈ a75 69.25). Không phải lời.
- **history depth >8**: bão hoà (h12 = h8).
- **WTS overhead->repeat**: flat (+0.02).
- ĐÚNG & QUAN TRỌNG: **resolution PHẢI 1280×720** (704×1216 sai tỉ lệ làm tụt điểm nặng, +2.35 khi sửa).

## 8. CẦN CHẠY TIẾP (THEO THỨ TỰ)
**Bước 1 — RE-GATE repeat_anchor đã sửa (xác nhận motion + Pareto):**
```
cd /workspace && conda activate wan22
rm -rf /workspace/experiments/wan_ticft/basemode_gate
CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_basemode_gate.sh
```
Đọc `basemode_gate/eval_pixel.csv` + REAL-FID. So `mf_video` vs `mf_repeat`(mới). **KIỂM TRA video `basemode_gate/mf_a75_h4_repeat_anchor/videos/*.avi` ĐÃ CÓ MOTION** (frame đầu vs cuối khác nhau — KHÔNG còn static như bug cũ). Dựng montage bằng Python311 ngoài container.

**Bước 2 — nếu repeat_anchor (fixed) Pareto (REAL-FID <= mf_video, PSNR/SSIM giữ, CÓ motion):**
```
CUDA_VISIBLE_DEVICES=2 MAX_HISTORY=8 TAG=mf_repeat BASE_MODE=repeat_anchor \
bash /workspace/wan_ticft/evaluation/build_multiframe_wts_candidate.sh
```
-> submit `experiments/hybrid_multiframe_mf_repeat_wts_candidate`, so mf_h8=69.34. **THEO DÕI FVD** (foreground frozen có thể hại FVD — gate chỉ đo image-FID, không đo FVD).

**Bước 3 — nếu frozen foreground hại FVD/PSNR:** warp foreground theo OPTICAL FLOW thay vì freeze (pixel thật DI CHUYỂN đúng hướng, vẫn 1 bản, không ghost). Đây là method tiếp theo đúng bài.

**Bước 4 — lever lớn chưa khai thác:** nửa **WTS (27/71 = 38%)** luôn dùng object-layer, chưa tối ưu nghiêm túc (mới thử overhead->repeat = flat). Nếu WTS đang yếu thì cải thiện 38% test có thể nhảy điểm.

## 9. FOLDER QUAN TRỌNG (GIỮ — KHÔNG XOÁ)
- `experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate` — a75 winner 69.24
- `experiments/hybrid_multiframe_mf_h8_wts_candidate` — BEST 69.34
- `experiments/wan_ticft/cnn_ltxmix65_bdd_test` — BDD base cho multiframe (base_mode=video)
- `experiments/wan_ticft/depth_layered_a75_bdd_test` — a75 BDD render
- `experiments/wan_ticft/wts_overhead_object_layer_vehicle_medium_test` — WTS 27 mẫu
- `experiments/wan_ticft/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/` — a75 val (videos/ + report.csv), dùng làm baseline gate
- `experiments/wan_ticft/frames_balanced_val_125/val/<sid>/target/` — GT THẬT cho REAL-FID gate
- `experiments/wan_ticft/ltxv_bdd25_gate/` — val manifest `selected_manifest.jsonl` + base `report.csv`
- `experiments/wan_ticft/depth_layered_models/` — cache model depth (DPT)
- `experiments/wan_ticft/ltx2b_chunked_1280x720_bdd_test` — chunk test (chunk đã chết, giữ phòng hờ)

## 10. ẢNH/VIDEO ĐỂ XEM (cách xem)
- Video val: `basemode_gate/mf_a75_h4_video/videos/<sid>.avi` (có ghost) và `..._repeat_anchor/videos/<sid>.avi`.
- Dựng montage so sánh: chạy Python311 ngoài container, cv2.VideoCapture đọc .avi (lấy frame cuối), cv2.imread GT png, hstack [video|repeat|GT], imwrite png, rồi mở xem. (Ví dụ script cũ đã dùng: đọc 3 sample val_bdd_pc_5k_video1407/2149/3237_phase0.)
- Để kiểm "có motion": đọc frame 0 và frame cuối của 1 .avi, nếu giống hệt = static (bug); khác nhau = có motion (đúng).

## 11. SCRIPTS CHÍNH (trong /workspace/wan_ticft/evaluation/)
- `compute_real_fid.py` — REAL-FID gate (pytorch-fid vs GT, có degrade option).
- `run_basemode_gate.sh` — gate video vs repeat_anchor (val BDD24).
- `materialize_depth_layered_multiframe.py` / `_test.py` — method chính (đã có `--base_mode video|repeat_anchor`, biến `fillbase`).
- `build_multiframe_wts_candidate.sh` — build submission từ multiframe (env MAX_HISTORY/HISTORY_STRIDE/BASE_MODE/CHUNK_DIR/TAG/CPU_THREADS).
- `materialize_depth_layered_renderer.py` / `_test.py` — a75 single-frame gốc (chứa helpers + compose/compose_adaptive + da3 depth backend + adaptive_blend/temporal_decay flags CHƯA dùng).
- `build_depth_layered_a75_wts_overhead_candidate.sh` — build a75 winner gốc.
- `run_real_fid_validate.sh`, `run_degrade_fid_sweep.sh`, `run_degrade_fid_fine.sh` — các gate đã dùng.

## 12. GHI CHÚ KỸ THUẬT
- `compose(base, rendered, confidence, alpha)`: out = base*(1-w) + rendered*w, w=clip(alpha*conf,0,0.85). Vùng conf thấp -> ra base.
- `render_confidence` = coverage * (1-actor) * photometric(exp(-|rendered-base|/sigma)) * pose_quality, eroded+blur+EMA. Disocclusion bị splat dilation lấp -> coverage=1 nhưng pixel xé (drag).
- `zbuffer_splat(anchor, depth, static_mask, pose, intrinsic, point_stride, splat_radius)`: forward splat 3D, trả (rendered, coverage).
- multiframe `compose_pose(p_hi_anchor, p_anchor_target)` ghép 2 pose (Rodrigues): R=R2@R1, t=R2@t1+t2.
- a75 params: alpha 0.75, focal_scale 0.9, point_stride 2, splat_radius 1, min_matches 20, min_inliers 12, photo_sigma 32, segment_batch 4, pose_smoothing_window 7, filled_pose_scale 0.35, confidence_ema 0.65.
