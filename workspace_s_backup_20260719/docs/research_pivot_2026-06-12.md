# Track 5 Research Pivot - 2026-06-12

## Stop Decision

Do not continue output-space optical-flow/refiner experiments for BDD.

Evidence:

- Refiner/router improves BDD25 PSNR and SSIM.
- Direct outputs visibly contain ghosting, stretched geometry, and duplicated edges.
- Blending Wan with the router improves pixel metrics monotonically only by
  progressively introducing the same artifacts.
- Previous leaderboard submissions already showed that improving PSNR/SSIM
  through pixel warping can reduce the final score through worse LPIPS/FID/FVD.

Keep:

- WTS views: `region_norm_medium`.
- BDD fallback: official Wan2.2.
- Refiner/router: validation diagnostic only.

## Paper Review

### Rejected for immediate experimentation

- FAR: native multi-context prediction, but released checkpoints are low
  resolution and trained on UCF101/BAIR/DMLab. Domain adaptation is expensive.
- FramePack: strong long-context/drift design, but the released product path is
  primarily image-to-video rather than traffic-history-conditioned prediction.
- SG-I2V: zero-shot trajectory control, but its released implementation targets
  SVD and bounding-box trajectories rather than the existing Wan stack.

### Selected direction: ATI / Wan-Move

Use trajectories extracted from observed history as motion conditioning inside
Wan. This preserves a generative decoder and avoids directly warping output
pixels.

ATI is selected first because it provides:

- an official Wan2.1 implementation;
- motion extraction tools from videos;
- camera, object, and local trajectory control;
- a clear four-sample feasibility path.

## Strict Experiment Gate

1. Prepare four BDD history clips.
2. Extract and inspect trajectories.
3. Only if trajectories correctly represent ego-camera and object motion,
   download/run ATI 14B on the same four samples.
4. Stop immediately if ATI has scene mismatch, weak motion fidelity, or worse
   visual consistency than official Wan on most samples.
5. No seed sweep, alpha sweep, or full-test generation before passing the gate.

## ATI Gate Result

ATI was stopped before checkpoint download.

- Dense CoTracker trajectories were successfully extracted from four BDD
  history clips.
- Visual inspection showed long and inconsistent background trajectories that
  would likely deform roads, buildings, and vehicles.
- ATI's official motion-transfer extractor also requires at least 120 observed
  frames, while Track 5 histories contain only 12--45 frames.

## Next Gate: VACE Region-Control Refinement

Test one BDD sample using Wan2.1 VACE 1.3B:

- `region_norm_medium` prediction as structural control video;
- final history frame as reference image;
- 49 frames, 10 inference steps, one seed.

Stop VACE immediately if this one output changes the scene, weakens realism, or
retains the region-flow artifacts.

## VACE Gate Result

VACE was stopped after one BDD sample.

- The generated result closely copied the region-control video.
- It retained the weak motion and visible blur/ghosting.
- There was no clear perceptual-quality improvement that justified another
  sample or parameter sweep.

## Next Gate: Cosmos-Predict2.5 Video2World

Cosmos-Predict2.5-2B Video2World is the next candidate because it:

- directly accepts an observed video plus text;
- predicts future world states rather than post-processing warped frames;
- targets Physical AI and autonomous-vehicle scenarios;
- outputs 720P video at 16 FPS.

The first gate only audits the environment, prepares four BDD history clips,
and clones official source. Do not install its environment or download
checkpoints until that audit passes.

## Cosmos-Predict2.5 Gate Result

Cosmos-Predict2.5 was stopped before environment installation and checkpoint
download.

- The official Video2World path hardcodes two latent conditional frames.
- Two latent frames correspond to only the final five pixel frames from the
  supplied history video.
- Cosmos 3 also accepts at most five input video frames, and its smallest
  released model is 16B.
- Neither option justifies its setup and inference cost for this task.

## Next Gate: LTX-Video 2B Video Extension

Test LTX-Video 2B distilled because its official pipeline supports:

- conditioning on an entire short video segment;
- forward video extension;
- 121-frame generation below 720p;
- a lightweight 2B checkpoint suitable for a strict one-sample gate.

The gate is:

1. Prepare four BDD history segments using the required `8n+1` frame length.
2. Clone source and audit resources without downloading checkpoints.
3. Run exactly one 2B distilled continuation.
4. Stop immediately if scene identity or camera geometry is worse than Wan.

## LTX-Video Gate Result

LTX-Video passed the BDD validation gate and is the current candidate for
replacing the official Wan BDD branch.

- 24/25 BDD validation samples completed; one source video could not be
  decoded consistently.
- On a dense 40-frame future evaluation:
  - LTX: `16.9166 PSNR`, `0.7547 SSIM`, `24.2222 AbsDiff`
  - repeat-last: `17.8261 PSNR`, `0.7251 SSIM`, `22.7663 AbsDiff`
  - official Wan: `16.0970 PSNR`, `0.6947 SSIM`, `26.1680 AbsDiff`
- LTX beat official Wan by per-sample PSNR on 19/24 samples.
- LTX and repeat-last each won 12/24 samples.
- Visual review showed coherent ego motion and substantially fewer scene
  hallucinations than Wan. The main failures were nearly static scenes and
  excessive lighting/object displacement.

Decision:

- Do not route by event phase; every phase contains both LTX wins and losses.
- Do not replace LTX with repeat-last for the final-score candidate because
  repeat-last is likely to weaken LPIPS/FID/FVD.
- Run a four-sample BDD test smoke using input history PNGs directly.
- If the smoke passes, build a separate `LTX BDD + region_norm_medium WTS`
  submission while retaining the submitted Wan/region hybrid as fallback.

## LTX BDD Test Generation

Full BDD test generation completed successfully.

- BDD samples: `44/44`
- Generated BDD frames: `2892`
- Every sample contains exactly `target_N` frames named from `0.png`.
- Every frame is readable at `1280x720`.
- Contact sheets: `44/44`
- No zero-byte files, missing indices, freezes, or scene collapses were found.

Visual review found a recurring semi-transparent rectangular artifact near an
edge of several LTX outputs. This is a real perceptual risk, but the outputs
still preserve scene identity and traffic motion better than the official Wan
BDD candidate. Build and submit LTX as a separate experimental candidate:

- BDD: LTX-Video 2B
- WTS fixed/vehicle: `region_norm_medium`

Keep the submitted Wan/region hybrid (`67.3044`) as the fallback.

## LTX Early-Frame Transition Gate

Two inexpensive last-condition-to-LTX transitions were evaluated on the
24-sample dense BDD validation set:

- LTX: `16.9166 PSNR`, `0.7547 SSIM`, `24.2222 AbsDiff`
- transition 8: `17.2566 PSNR`, `0.7617 SSIM`, `24.1481 AbsDiff`
- transition 16: `17.6807 PSNR`, `0.7699 SSIM`, `23.3319 AbsDiff`
- repeat-last: `17.8261 PSNR`, `0.7251 SSIM`, `22.7663 AbsDiff`

Transition 16 won 14/24 samples and visual review found no new scene collapse
or obvious transition jump. Build one separate transition-16 submission. Do
not test further transition lengths unless leaderboard evidence supports it.

The transition-16 submission scored `68.2611`, improving every leaderboard
metric over raw LTX (`67.4521`). This confirms that early-frame anchoring is
beneficial.

LTX was also tested on four vehicle/IP validation samples. It beat Wan but was
substantially worse than `region_norm_medium`, so LTX remains BDD-only.

A single static-scene router gate routed the lowest-motion 20% of BDD samples
to `region_norm_medium`:

- LTX transition 16: `17.6807 PSNR`, `0.7699 SSIM`, `23.3319 AbsDiff`
- static router: `17.8169 PSNR`, `0.7674 SSIM`, `23.1912 AbsDiff`

The small PSNR/AbsDiff gain came with lower SSIM and would likely weaken
perceptual/video metrics. Reject the static router and do not build a
submission from it.

A single LTX multi-keyframe gate added the last observed frame as a weak
future anchor (`strength=0.25`, future offset 16) on four mixed-phase BDD
samples:

- future anchor: `16.8477 PSNR`, `0.8060 SSIM`, `23.2070 AbsDiff`
- raw LTX: `17.5213 PSNR`, `0.8426 SSIM`, `20.4767 AbsDiff`
- LTX transition 16: `18.4482 PSNR`, `0.8576 SSIM`, `19.7169 AbsDiff`

The anchor degraded every metric and visually caused incorrect persistence,
warping, and scene changes. Reject future-anchor conditioning without further
strength or offset sweeps.

## Sources

- https://github.com/bytedance/ATI
- https://github.com/ali-vilab/Wan-Move
- https://github.com/showlab/FAR
- https://github.com/lllyasviel/FramePack
- https://github.com/Kmcode1/SG-I2V
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://github.com/Lightricks/LTX-Video
