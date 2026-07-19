# Track 5 High-Score Research Direction - 2026-06-13

## Target And Reality Check

Current best submission:

- final score: `68.2611`
- PSNR: `18.2169`
- SSIM: `0.6114`
- LPIPS: `0.3393`
- CLIP: `0.9348`
- FID: `50.2430`
- FVD: `32.1304`

A score above 80 requires a large improvement across fidelity and generative
metrics. Post-processing the current LTX output cannot provide that jump.
Score 90 should be treated as a research target, not a credible short-term
promise, because the future is ambiguous and the hidden test set evaluates
only one realized future.

## Main Finding

The current systems choose between two bad extremes:

- full-frame video generation preserves realism but hallucinates scene layout;
- optical-flow prediction preserves pixels but blurs, freezes, and deforms.

The new direction is to stop generating the complete future frame.

## Proposed Architecture: Geometry-Residual Forecasting

```text
history video
  -> dynamic-aware 3D reconstruction and camera estimation
  -> robust camera and actor trajectory extrapolation
  -> geometry-rendered future scaffold
  -> residual video diffusion for dynamic regions and disocclusions
  -> exact-length 1280x720 output
```

### Geometry branch

Start with V-DPM because it estimates per-point 3D motion in addition to camera
and geometry. Use VGGT-Omega as the fallback geometry backbone if V-DPM cannot
handle the histories. Extrapolate the observed camera and scene trajectories,
then render a future background scaffold with point or Gaussian splatting.

This branch should preserve road layout, parked vehicles, buildings, and
camera motion much more accurately than affine or homography warping.

### Dynamic branch

Track pedestrian and vehicle regions through history. Predict their future
motion using observed trajectories plus structured caption attributes such as:

- vehicle speed and maneuver;
- pedestrian crossing direction and speed;
- event phase;
- relative actor positions.

### Residual generator

Fine-tune a video model to predict only:

- dynamic-object residuals;
- disoccluded pixels;
- corrections to uncertain geometry regions.

The geometry-rendered scaffold remains unchanged in confident static regions.
This should retain LTX-like realism while avoiding full-scene hallucination.

## Data Strategy

Do not train on only 20 or 500 clips.

Available indexed data:

- train rows: `14,846`
- validation rows: `6,215`
- BDD train rows: `12,151`
- WTS train rows: `2,695`

Build rolling windows from complete source videos instead of using only one
clip per phase boundary. Randomize:

- observed history length;
- future horizon from 51 to 120 frames;
- temporal stride;
- phase and view type.

Train view-specific branches:

- BDD ego view: geometry-residual model;
- WTS vehicle/IP: geometry-residual model after BDD validation;
- WTS fixed/overhead: static background plus object-only residual model.

## Losses

Use a metric-aligned combination:

- Charbonnier/L1 and SSIM for pixel fidelity;
- LPIPS on full frames and dynamic crops;
- temporal flow and feature consistency for FVD;
- background identity loss outside uncertain/dynamic masks;
- caption/CLIP alignment at low weight;
- stronger residual penalties in confident geometry regions.

## Strict Gates

### Gate 1: Spatial-composite oracle ceiling

Measure block-wise oracle compositing between LTX transition16 and
region-normalized prediction on the existing 24-sample dense BDD validation
set.

Proceed only if block-64 oracle improves over LTX transition16 by at least:

- `+1.5 PSNR`, or
- `+0.03 SSIM` with materially lower AbsDiff.

### Gate 2: V-DPM geometry smoke

Run four complementary BDD histories: two where region prediction strongly
wins and two where LTX strongly wins. Render only the future static scaffold.
Proceed only if static-region fidelity clearly beats affine/region prediction
without severe holes or camera drift.

V-DPM direct-mask result:

- V-DPM correctly distinguished the strongest ego-motion sample from the three
  lower-motion samples.
- Motion overlays were not spatially selective enough: large sky, foliage,
  road, and background regions were highlighted alongside true actors.
- Reject V-DPM as a direct compositor mask. Retain its 3D features only as a
  possible input to a learned selector.

### Gate 2B: Learned spatial selector

Train a lightweight block selector using only observable features from:

- history frames;
- region prediction;
- LTX transition16 prediction;
- spatial position and horizon.

Evaluate with two-fold cross-validation split by sample. Proceed only if the
held-out learned selector beats both region and LTX transition16, rather than
merely fitting the oracle labels.

The lightweight block-feature selector passed:

- region: `18.2520` PSNR, `0.7557` SSIM, `21.5902` AbsDiff;
- LTX transition16: `17.6807` PSNR, `0.7699` SSIM, `23.3319` AbsDiff;
- learned selector: `18.3776` PSNR, `0.7759` SSIM, `20.9748` AbsDiff;
- selector accuracy: `0.6909`.

This proves that candidate preference is partially predictable without ground
truth, but the learned selector remains far below the `19.7591` block-oracle
ceiling. The next bounded gate is a convolutional continuous-mask selector.
Stop the selector path if that model does not materially close the gap.

The convolutional selector passed two-fold sample-level cross-validation:

- region: `18.2520` PSNR, `0.7557` SSIM, `21.5902` AbsDiff;
- LTX transition16: `17.6807` PSNR, `0.7699` SSIM, `23.3319` AbsDiff;
- CNN selector: `19.0539` PSNR, `0.8019` SSIM, `19.5508` AbsDiff.

The CNN beat LTX transition16 on all `24/24` samples, beat region on `19/24`,
and was never worse than both candidates on any sample. Promote it to a
separate test candidate, but inspect temporal seams and blending artifacts
before packaging or submission.

### Gate 3: Residual model smoke

Train a small residual model on a limited but representative BDD subset.
Proceed only if it improves both pixel fidelity and perceptual quality over
the geometry scaffold and LTX transition16.

### Gate 4: Full training

Only after all prior gates pass, materialize rolling-window training data and
train the full geometry-residual model.

## Stop Decisions

- Stop LTX LoRA smoke: negligible improvement and severe pipeline mismatch.
- Stop transition-curve and color-correction sweeps if the single gates fail.
- Do not run more full-frame diffusion seed sweeps.

## Research Basis

- AI City Track 5 evaluates fidelity, perceptual similarity, semantic
  alignment, and temporal coherence together.
- VGGT-Omega supports efficient feed-forward reconstruction for static and
  dynamic scenes and predicts camera/depth/point geometry.
- Diffusion Forcing supports history-conditioned sequence generation with
  flexible conditioning.
- DepthCrafter demonstrates temporally consistent video depth prediction.

Sources:

- https://www.aicitychallenge.org/2026-track5/
- https://github.com/eldar/vdpm
- https://github.com/facebookresearch/vggt-omega
- https://vggt-omega.github.io/
- https://github.com/facebookresearch/vggt
- https://github.com/buoyancy99/diffusion-forcing
- https://github.com/Tencent/DepthCrafter
