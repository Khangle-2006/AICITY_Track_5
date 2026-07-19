# LTX Traffic Prompt Templates

These prompts follow the LTX-2.3 prompting guidance: detailed single-paragraph
shot descriptions, present-tense action, explicit camera behavior, lighting,
surface detail, and motion continuity. They are written for traffic-video
continuation where geometry and full-frame sharpness matter more than cinematic
bokeh.

## BDD Dashcam Continuation

The video is continuous realistic dashcam footage from a moving vehicle on an
urban road. Continue the observed scene for the next few seconds with the same
camera viewpoint, same focal length, same exposure, same weather, and same
lighting. The ego vehicle moves smoothly forward with natural low-amplitude
dashcam vibration; the road perspective expands consistently from the exact last
observed frame. Preserve the lane geometry, curb lines, sidewalks, buildings,
trees, traffic lights, parked vehicles, moving vehicles, pedestrians, shadows,
reflections, and background layout. Vehicles ahead keep realistic speed and lane
position, with no sudden jumps, no new vehicles, and no disappearing objects.
Pedestrians, if visible, continue their current motion naturally. Keep the whole
frame sharp and documentary-realistic: crisp lane markings, detailed asphalt
texture, readable object boundaries, stable road edges, clear vehicle contours,
natural motion blur only from real movement, no dreamy softness, no cinematic
bokeh, no stylization. The shot is a single unbroken traffic-camera continuation
with physically plausible motion and temporally stable details.

## BDD Dashcam Native

Continue the same real dashcam recording from the final observed frame. Keep the
existing lens, exposure, color response, windshield reflections, sensor noise,
compression texture, motion blur, and low-detail distant background exactly
consistent with the input video. Preserve the road layout, lane markings, curbs,
buildings, parked cars, traffic lights, shadows, visible pedestrians, vehicle
scale, and camera viewpoint without redesigning or beautifying the scene. The
near future should look like ordinary unedited driving footage from the same
camera, not a cinematic render or a polished generated image. Only continue the
already observed ego motion and actor motion; do not introduce new objects,
remove objects, sharpen surfaces artificially, repaint textures, change
lighting, or clean up the image.

## WTS Fixed/Overhead Continuation

The video is continuous realistic fixed traffic-surveillance footage from the
same elevated camera. Continue the observed scene with a locked camera, fixed
perspective, fixed background, stable exposure, and unchanged lighting. The road,
crosswalk, lane markings, sidewalks, buildings, parked cars, shadows, and static
background remain stable and sharp. Only already-visible pedestrians and
vehicles continue moving along their current trajectories with realistic speed
and smooth temporal motion. Do not introduce new people or cars. Do not change
the camera angle, zoom, color tone, or background geometry. Preserve crisp
object boundaries, clear road texture, stable contours, and documentary
surveillance-video realism with no blur, no hallucinated objects, no stylized
cinematic look, and no flicker.

## Negative Prompt

blurry, soft focus, dreamy, cinematic bokeh, shallow depth of field, smeared
road texture, unstable lane markings, warped buildings, flickering background,
jittery motion, sudden camera jump, new cars, new pedestrians, disappearing
objects, duplicated vehicles, distorted wheels, melted road, stylized animation,
cartoon, painterly, low detail, low quality
