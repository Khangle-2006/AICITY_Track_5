"""Depth-aware static-background renderer over a frozen video-prediction base.

The final observed frame supplies sharp texture and relative depth. Static ORB
matches to each base prediction estimate a PnP camera pose. A z-buffered 3D
forward splat renders only non-actor history pixels into the base view. Actor
regions, disocclusions, unreliable poses, and uncovered pixels remain exactly
the base prediction. This module is inference-only and never uses targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps


ACTOR_NAMES = {"person", "bicycle", "bus", "car", "motorbike", "motorcycle", "horse"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2_module, paths: list[str]):
    return [image for raw in paths if (image := read_rgb(cv2_module, Path(raw))) is not None]


def load_base_frames(cv2_module, row: dict, video: Path):
    offsets = map_target_offsets(row, video_fps(cv2_module, video))
    frame_map = read_video_rgb_frames(cv2_module, video, offsets)
    return [frame_map[index] for index in offsets if index in frame_map]


def load_full_video_frames(cv2_module, video: Path):
    capture = cv2_module.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB))
    capture.release()
    return frames


def write_video(cv2_module, frames, path: Path, fps: float, prepend_guard: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2_module.VideoWriter(str(path), cv2_module.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    output_frames = [frames[0], *frames] if prepend_guard else frames
    for frame in output_frames:
        writer.write(cv2_module.cvtColor(frame, cv2_module.COLOR_RGB2BGR))
    writer.release()


def compute_raft_flow(models, img1, img2):
    # Load RAFT model lazily
    if not hasattr(models, 'raft_model'):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        models.raft_weights = Raft_Large_Weights.DEFAULT
        models.raft_model = raft_large(weights=models.raft_weights).eval().to(models.device)
        models.raft_transforms = models.raft_weights.transforms()
        
    h, w = img1.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    if h_pad > 0 or w_pad > 0:
        img1_padded = np.pad(img1, ((0, h_pad), (0, w_pad), (0, 0)), mode='edge')
        img2_padded = np.pad(img2, ((0, h_pad), (0, w_pad), (0, 0)), mode='edge')
    else:
        img1_padded = img1
        img2_padded = img2

    t1 = torch.from_numpy(img1_padded).permute(2, 0, 1)[None].to(models.device)
    t2 = torch.from_numpy(img2_padded).permute(2, 0, 1)[None].to(models.device)
    t1_t, t2_t = models.raft_transforms(t1, t2)
    
    with torch.no_grad():
        flow = models.raft_model(t1_t, t2_t)[-1][0].permute(1, 2, 0).cpu().numpy()
        
    flow = flow[:h, :w]
    return flow


class PerceptionModels:
    def __init__(self, device: torch.device, cache_dir: Path):
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        self.device = device
        self.cache_dir = cache_dir
        self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=cache_dir)
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas", cache_dir=cache_dir, torch_dtype=torch.float32
        ).eval().to(device)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.segment_model = deeplabv3_resnet50(weights=weights).eval().to(device)
        self.segment_transform = weights.transforms()
        categories = weights.meta["categories"]
        self.actor_ids = {index for index, name in enumerate(categories) if name.lower() in ACTOR_NAMES}

    def load_superpoint(self):
        if not hasattr(self, 'superpoint'):
            from lightglue import LightGlue, SuperPoint
            self.superpoint = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.lightglue = LightGlue(features='superpoint').eval().to(self.device)

    def load_depth_anything(self, scale="small"):
        repo_map = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        repo_id = repo_map.get(scale, "depth-anything/Depth-Anything-V2-Small-hf")
        attr_name = f"depth_anything_{scale}"
        attr_proc = f"depth_anything_processor_{scale}"
        if not hasattr(self, attr_name):
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            setattr(self, attr_proc, AutoImageProcessor.from_pretrained(repo_id, cache_dir=self.cache_dir))
            setattr(self, attr_name, AutoModelForDepthEstimation.from_pretrained(
                repo_id, cache_dir=self.cache_dir
            ).eval().to(self.device))
        return getattr(self, attr_proc), getattr(self, attr_name)

    def load_yolo(self):
        if not hasattr(self, 'yolo_model'):
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n-seg.pt").to(self.device)

    def load_sam2(self):
        if not hasattr(self, 'sam2_model'):
            from transformers import Sam2VideoProcessor, Sam2VideoModel
            self.sam2_processor = Sam2VideoProcessor.from_pretrained(
                "facebook/sam2.1-hiera-tiny", cache_dir=self.cache_dir
            )
            self.sam2_model = Sam2VideoModel.from_pretrained(
                "facebook/sam2.1-hiera-tiny", cache_dir=self.cache_dir, torch_dtype=torch.float32
            ).eval().to(self.device)

    @torch.inference_mode()
    def depth(self, rgb: np.ndarray, model_type="midas") -> np.ndarray:
        if model_type.startswith("depthanything"):
            scale = "small"
            if "base" in model_type:
                scale = "base"
            elif "large" in model_type:
                scale = "large"
            processor, model = self.load_depth_anything(scale=scale)
            inputs = processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            prediction = model(**inputs).predicted_depth[:, None]
            prediction = torch.nn.functional.interpolate(
                prediction, size=rgb.shape[:2], mode="bicubic", align_corners=False
            )[0, 0]
            inverse = prediction.float().cpu().numpy()
            low, high = np.percentile(inverse[np.isfinite(inverse)], [2.0, 98.0])
            normalized = np.clip((inverse - low) / max(high - low, 1e-6), 0.0, 1.0)
            return (1.0 / (0.12 + normalized)).astype(np.float32)
        else:
            inputs = self.depth_processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            prediction = self.depth_model(**inputs).predicted_depth[:, None]
            prediction = torch.nn.functional.interpolate(
                prediction, size=rgb.shape[:2], mode="bicubic", align_corners=False
            )[0, 0]
            inverse = prediction.float().cpu().numpy()
            low, high = np.percentile(inverse[np.isfinite(inverse)], [2.0, 98.0])
            normalized = np.clip((inverse - low) / max(high - low, 1e-6), 0.0, 1.0)
            return (1.0 / (0.12 + normalized)).astype(np.float32)

    @torch.inference_mode()
    def actor_masks(self, frames: list[np.ndarray], batch_size: int, model_type="deeplabv3") -> list[np.ndarray]:
        if model_type == "sam2":
            self.load_sam2()
            self.load_yolo()
            
            anchor = frames[0]
            height, width = anchor.shape[:2]
            anchor_bgr = cv2.cvtColor(anchor, cv2.COLOR_RGB2BGR)
            results = self.yolo_model(anchor_bgr, verbose=False)[0]
            
            boxes = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0].cpu().item())
                    cls_name = self.yolo_model.names[cls_id].lower()
                    if cls_name in ACTOR_NAMES:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        boxes.append(xyxy)
                        
            if not boxes:
                return [np.zeros((height, width), dtype=np.float32) for _ in frames]
                
            session = self.sam2_processor.init_video_session(frames, inference_device=str(self.device))
            obj_ids = list(range(len(boxes)))
            input_boxes = [boxes]
            self.sam2_processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=obj_ids,
                input_boxes=input_boxes
            )
            
            frame_masks = {i: np.zeros((height, width), dtype=np.uint8) for i in range(len(frames))}
            for out in self.sam2_model.propagate_in_video_iterator(session, start_frame_idx=0):
                frame_idx = out.frame_idx
                pred_masks = out.pred_masks
                
                res_masks = self.sam2_processor.post_process_masks(
                    [pred_masks],
                    original_sizes=[[height, width]],
                    binarize=True
                )[0]
                
                if res_masks.shape[0] > 0:
                    combined_mask = (res_masks.sum(dim=(0, 1)) > 0).cpu().numpy().astype(np.uint8) * 255
                    frame_masks[frame_idx] = combined_mask
                    
            masks = []
            for i in range(len(frames)):
                mask = frame_masks[i]
                mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
                mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)
                masks.append(np.clip(mask, 0.0, 1.0))
            return masks
        elif model_type == "yolov8":
            self.load_yolo()
            masks = []
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = self.yolo_model(frame_bgr, verbose=False)[0]
                if results.masks is not None:
                    actor_indices = []
                    for idx, cls_id in enumerate(results.boxes.cls.cpu().numpy()):
                        cls_name = self.yolo_model.names[int(cls_id)].lower()
                        if cls_name in ACTOR_NAMES:
                            actor_indices.append(idx)
                    if actor_indices:
                        actor_masks_list = results.masks.data[actor_indices].cpu().numpy()
                        mask = (actor_masks_list.sum(axis=0) > 0.5).astype(np.uint8) * 255
                    else:
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
                mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)
                masks.append(np.clip(mask, 0.0, 1.0))
            return masks
        else:
            masks = []
            for start in range(0, len(frames), batch_size):
                batch = frames[start : start + batch_size]
                tensors = [self.segment_transform(torch.from_numpy(frame).permute(2, 0, 1)) for frame in batch]
                logits = self.segment_model(torch.stack(tensors).to(self.device))["out"]
                labels = logits.argmax(1).cpu().numpy()
                for label, frame in zip(labels, batch):
                    mask = np.isin(label, list(self.actor_ids)).astype(np.uint8) * 255
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
                    mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)
                    masks.append(np.clip(mask, 0.0, 1.0))
            return masks


def camera_matrix(width: int, height: int, focal_scale: float) -> np.ndarray:
    focal = focal_scale * max(width, height)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_pose(anchor, base, depth, actor_mask, intrinsic, min_matches: int, min_inliers: int, models=None, matcher="orb"):
    if matcher == "superpoint" and models is not None:
        models.load_superpoint()
        img_anchor = torch.from_numpy(anchor).permute(2, 0, 1).float().to(models.device) / 255.0
        img_base = torch.from_numpy(base).permute(2, 0, 1).float().to(models.device) / 255.0
        
        with torch.no_grad():
            feats_anchor = models.superpoint({'image': img_anchor[None]})
            feats_base = models.superpoint({'image': img_base[None]})
            
            # Guard against zero or insufficient keypoints
            if feats_anchor['keypoints'].shape[1] < min_matches or feats_base['keypoints'].shape[1] < min_matches:
                return None, {"matches": 0, "inliers": 0, "reprojection": 999.0}
                
            matches_dict = models.lightglue({'image0': feats_anchor, 'image1': feats_base})
            
            from lightglue.utils import rbd
            feats_anchor, feats_base, matches_dict = [rbd(x) for x in [feats_anchor, feats_base, matches_dict]]
            
        key_anchor = feats_anchor['keypoints'].cpu().numpy()
        key_base = feats_base['keypoints'].cpu().numpy()
        matches = matches_dict['matches'].cpu().numpy()
        
        good_matches = []
        for m in matches:
            idx0, idx1 = m
            x, y = key_anchor[idx0]
            xi, yi = int(round(x)), int(round(y))
            if actor_mask[min(max(yi, 0), actor_mask.shape[0] - 1), min(max(xi, 0), actor_mask.shape[1] - 1)] < 0.1:
                good_matches.append((key_anchor[idx0], key_base[idx1]))
                
        if len(good_matches) < min_matches:
            return None, {"matches": len(good_matches), "inliers": 0, "reprojection": 999.0}
            
        image_points = np.float32([m[1] for m in good_matches])
        pixels = np.float32([m[0] for m in good_matches])
        px = np.clip(np.rint(pixels[:, 0]).astype(int), 0, depth.shape[1] - 1)
        py = np.clip(np.rint(pixels[:, 1]).astype(int), 0, depth.shape[0] - 1)
        z = depth[py, px]
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        object_points = np.stack([(pixels[:, 0] - cx) * z / fx, (pixels[:, 1] - cy) * z / fy, z], axis=1)
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            intrinsic,
            None,
            iterationsCount=300,
            reprojectionError=5.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        inlier_count = len(inliers) if ok and inliers is not None else 0
        if not ok or inlier_count < min_inliers:
            return None, {"matches": len(good_matches), "inliers": inlier_count, "reprojection": 999.0}
        projected, _ = cv2.projectPoints(object_points[inliers[:, 0]], rotation, translation, intrinsic, None)
        error = np.linalg.norm(projected[:, 0] - image_points[inliers[:, 0]], axis=1).mean()
        return (rotation, translation), {"matches": len(good_matches), "inliers": inlier_count, "reprojection": float(error)}
    else:
        anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_RGB2GRAY)
        base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=3000, fastThreshold=8)
        key_anchor, desc_anchor = orb.detectAndCompute(anchor_gray, None)
        key_base, desc_base = orb.detectAndCompute(base_gray, None)
        if desc_anchor is None or desc_base is None:
            return None, {"matches": 0, "inliers": 0, "reprojection": 999.0}
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_anchor, desc_base, k=2)
        good = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            first, second = pair
            x, y = key_anchor[first.queryIdx].pt
            xi, yi = int(round(x)), int(round(y))
            if first.distance < 0.72 * second.distance and actor_mask[min(max(yi, 0), actor_mask.shape[0] - 1), min(max(xi, 0), actor_mask.shape[1] - 1)] < 0.1:
                good.append(first)
        if len(good) < min_matches:
            return None, {"matches": len(good), "inliers": 0, "reprojection": 999.0}
        image_points = np.float32([key_base[item.trainIdx].pt for item in good])
        pixels = np.float32([key_anchor[item.queryIdx].pt for item in good])
        px = np.clip(np.rint(pixels[:, 0]).astype(int), 0, depth.shape[1] - 1)
        py = np.clip(np.rint(pixels[:, 1]).astype(int), 0, depth.shape[0] - 1)
        z = depth[py, px]
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        object_points = np.stack([(pixels[:, 0] - cx) * z / fx, (pixels[:, 1] - cy) * z / fy, z], axis=1)
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            intrinsic,
            None,
            iterationsCount=300,
            reprojectionError=5.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        inlier_count = len(inliers) if ok and inliers is not None else 0
        if not ok or inlier_count < min_inliers:
            return None, {"matches": len(good), "inliers": inlier_count, "reprojection": 999.0}
        projected, _ = cv2.projectPoints(object_points[inliers[:, 0]], rotation, translation, intrinsic, None)
        error = np.linalg.norm(projected[:, 0] - image_points[inliers[:, 0]], axis=1).mean()
        return (rotation, translation), {"matches": len(good), "inliers": inlier_count, "reprojection": float(error)}


def raw_pose_quality(diagnostic: dict) -> float:
    if diagnostic["inliers"] <= 0 or diagnostic["reprojection"] >= 999.0:
        return 0.0
    return float(min(1.0, diagnostic["inliers"] / 40.0) * np.exp(-diagnostic["reprojection"] / 4.0))


def smooth_pose_sequence(poses, qualities, window: int, filled_scale: float):
    valid = np.array([pose is not None and quality >= 0.05 for pose, quality in zip(poses, qualities)])
    if valid.sum() < 2 or window <= 1:
        return poses, qualities, [False] * len(poses)
    indices = np.arange(len(poses), dtype=np.float64)
    valid_indices = indices[valid]
    vectors = np.full((len(poses), 6), np.nan, dtype=np.float64)
    for index, pose in enumerate(poses):
        if pose is not None:
            vectors[index, :3] = pose[0].reshape(3)
            vectors[index, 3:] = pose[1].reshape(3)
    interpolated = np.stack(
        [np.interp(indices, valid_indices, vectors[valid, component]) for component in range(6)], axis=1
    )
    window = min(window if window % 2 else window + 1, len(poses) if len(poses) % 2 else len(poses) - 1)
    radius = window // 2
    weights = np.arange(1, radius + 2, dtype=np.float64)
    weights = np.concatenate([weights, weights[-2::-1]])
    weights /= weights.sum()
    smoothed = np.stack(
        [np.convolve(np.pad(interpolated[:, component], radius, mode="edge"), weights, mode="valid") for component in range(6)],
        axis=1,
    )
    quality_values = np.asarray(qualities, dtype=np.float64)
    quality_interp = np.interp(indices, valid_indices, quality_values[valid])
    filled = (~valid).tolist()
    quality_interp[~valid] *= filled_scale
    output_poses = [(row[:3].reshape(3, 1), row[3:].reshape(3, 1)) for row in smoothed]
    return output_poses, quality_interp.tolist(), filled


def zbuffer_splat(anchor, depth, static_mask, pose, intrinsic, point_stride: int, splat_radius: int):
    height, width = anchor.shape[:2]
    ys, xs = np.mgrid[0:height:point_stride, 0:width:point_stride]
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    keep = static_mask[ys, xs] > 0.5
    xs, ys = xs[keep], ys[keep]
    z = depth[ys, xs]
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    points = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1).astype(np.float64)
    rotation, translation = pose
    matrix, _ = cv2.Rodrigues(rotation)
    transformed = points @ matrix.T + translation.reshape(1, 3)
    valid_z = transformed[:, 2] > 1e-4
    transformed, colors = transformed[valid_z], anchor[ys[valid_z], xs[valid_z]]
    u = np.rint(fx * transformed[:, 0] / transformed[:, 2] + cx).astype(np.int32)
    v = np.rint(fy * transformed[:, 1] / transformed[:, 2] + cy).astype(np.int32)
    projected_z = transformed[:, 2].astype(np.float32)
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            uu, vv = u + dx, v + dy
            valid = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            uu, vv, zz, cc = uu[valid], vv[valid], projected_z[valid], colors[valid]
            order = np.argsort(zz)[::-1]
            rendered[vv[order], uu[order]] = cc[order]
            zbuffer[vv[order], uu[order]] = zz[order]
    coverage = np.isfinite(zbuffer).astype(np.float32)
    return rendered, coverage


def render_confidence(
    base,
    rendered,
    coverage,
    base_actor,
    pose_quality: float,
    photo_sigma: float,
    previous_confidence,
    confidence_ema: float,
):
    difference = np.mean(np.abs(rendered.astype(np.float32) - base.astype(np.float32)), axis=2)
    photometric = np.exp(-difference / photo_sigma).astype(np.float32)
    confidence = coverage * (1.0 - base_actor) * photometric * pose_quality
    confidence = cv2.erode(confidence, np.ones((3, 3), np.uint8), iterations=1)
    confidence = cv2.GaussianBlur(confidence, (0, 0), 1.2)
    if previous_confidence is not None:
        confidence = confidence_ema * confidence + (1.0 - confidence_ema) * previous_confidence
    confidence *= 1.0 - base_actor
    return np.clip(confidence, 0.0, 1.0)


def compose(base, rendered, confidence, alpha: float, depth=None, use_depth_weighting=False):
    if use_depth_weighting and depth is not None:
        depth_horizon = np.percentile(depth, 75)
        weight_map = np.exp(-depth / max(depth_horizon, 1e-6))
        weight_map = np.clip(weight_map, 0.0, 1.0)[..., None]
        weight = np.clip(alpha * confidence * weight_map, 0.0, 0.85)
    else:
        weight = np.clip(alpha * confidence, 0.0, 0.85)[..., None]
    output = base.astype(np.float32) * (1.0 - weight) + rendered.astype(np.float32) * weight
    return np.clip(output, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument(
        "--anchor_override_root",
        type=Path,
        default=None,
        help="Optional root containing <sample_id>/frames/0003.png clean-anchor overrides.",
    )
    parser.add_argument(
        "--preserve_full_video",
        action="store_true",
        help="Render every consecutive base-video frame and preserve its timeline.",
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--horizon_decay", action="store_true")
    parser.add_argument("--horizon_decay_start", type=int, default=40)
    parser.add_argument("--horizon_decay_end", type=int, default=80)
    parser.add_argument("--horizon_decay_alpha", type=float, default=0.75)
    parser.add_argument("--horizon_decay_floor", type=float, default=0.25)
    parser.add_argument("--focal_scale", type=float, default=0.9)
    parser.add_argument("--point_stride", type=int, default=2)
    parser.add_argument("--splat_radius", type=int, default=1)
    parser.add_argument("--min_matches", type=int, default=20)
    parser.add_argument("--min_inliers", type=int, default=12)
    parser.add_argument("--photo_sigma", type=float, default=32.0)
    parser.add_argument("--segment_batch", type=int, default=4)
    parser.add_argument("--pose_smoothing_window", type=int, default=7)
    parser.add_argument("--filled_pose_scale", type=float, default=0.35)
    parser.add_argument("--confidence_ema", type=float, default=0.65)
    parser.add_argument("--clean", action="store_true")
    
    # New options
    parser.add_argument("--matcher", default="orb", choices=["orb", "superpoint"])
    parser.add_argument("--depth_model", default="midas", choices=["midas", "depthanything", "depthanything_small", "depthanything_base", "depthanything_large"])
    parser.add_argument("--segment_model", default="deeplabv3", choices=["deeplabv3", "yolov8", "sam2"])
    parser.add_argument("--disocclusion_inpainting", action="store_true")
    parser.add_argument("--alpha_decay", default="none", choices=["none", "linear", "exponential"])
    parser.add_argument("--alpha_decay_rate", type=float, default=0.95)
    parser.add_argument("--alpha_depth_weighting", action="store_true")

    args = parser.parse_args()

    cv2_module, _ = import_cv2_np()
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    (args.out_root / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    base_report = read_report(args.base_report)
    all_rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("view_type") == args.view_type and row["sample_id"] in base_report
    ]
    if args.sample_id:
        order = {sample_id: index for index, sample_id in enumerate(args.sample_id)}
        rows = [row for row in all_rows if row["sample_id"] in order]
        rows.sort(key=lambda row: order[row["sample_id"]])
        if len(rows) != len(args.sample_id):
            raise RuntimeError(f"requested {len(args.sample_id)} samples, found {len(rows)}")
    else:
        rows = all_rows[: args.limit]
    (args.out_root / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    device = torch.device("cuda")
    models = PerceptionModels(device, args.model_cache)
    variants = {f"depth_layered_a{int(round(alpha * 100)):02d}": alpha for alpha in args.alphas}
    if args.horizon_decay or args.alpha_decay != "none":
        variants["depth_layered_hdecay"] = args.horizon_decay_alpha
    reports = {name: [] for name in variants}
    diagnostics = []

    for position, row in enumerate(rows, start=1):
        base_video = base_report.get(row["sample_id"])
        conditions = load_images(cv2_module, (row.get("frames") or {}).get("condition", []))
        if base_video and args.preserve_full_video:
            bases = load_full_video_frames(cv2_module, base_video)
        else:
            bases = load_base_frames(cv2_module, row, base_video) if base_video else []
        if not conditions or not bases:
            print(f"[{position}/{len(rows)}] {row['sample_id']} | skip", flush=True)
            continue
        height, width = bases[0].shape[:2]
        anchor_source = conditions[-1]
        if args.anchor_override_root is not None:
            override_path = args.anchor_override_root / row["sample_id"] / "frames" / "0003.png"
            override = read_rgb(cv2_module, override_path)
            if override is None:
                raise RuntimeError(f"missing clean anchor override: {override_path}")
            anchor_source = override
        anchor = cv2.resize(anchor_source, (width, height), interpolation=cv2.INTER_AREA)
        bases = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in bases]
        
        # Load custom depth model if requested
        depth = models.depth(anchor, model_type=args.depth_model)
        
        # Load custom segmentation model if requested
        masks = models.actor_masks([anchor, *bases], args.segment_batch, model_type=args.segment_model)
        anchor_actor, base_actors = masks[0], masks[1:]
        static_mask = (anchor_actor < 0.1).astype(np.float32)
        intrinsic = camera_matrix(width, height, args.focal_scale)
        outputs = {name: [] for name in variants}
        raw_poses, pose_diagnostics, raw_qualities = [], [], []
        
        for base in bases:
            pose, pose_diag = estimate_pose(
                anchor, base, depth, anchor_actor, intrinsic, args.min_matches, args.min_inliers,
                models=models, matcher=args.matcher
            )
            raw_poses.append(pose)
            pose_diagnostics.append(pose_diag)
            raw_qualities.append(raw_pose_quality(pose_diag))
        poses, pose_qualities, pose_filled = smooth_pose_sequence(
            raw_poses, raw_qualities, args.pose_smoothing_window, args.filled_pose_scale
        )
        previous_confidence = None
        for frame_index, (base, base_actor, pose, pose_quality, filled) in enumerate(
            zip(bases, base_actors, poses, pose_qualities, pose_filled)
        ):
            pose_diag = pose_diagnostics[frame_index]
            if pose is None or pose_quality <= 0:
                rendered = base
                coverage = np.zeros((height, width), dtype=np.float32)
            else:
                rendered, coverage = zbuffer_splat(anchor, depth, static_mask, pose, intrinsic, args.point_stride, args.splat_radius)
                
                # Pro-active disocclusion inpainting using RAFT
                if args.disocclusion_inpainting:
                    disocclusion_mask = ((static_mask > 0.5) & (coverage < 0.1)).astype(np.float32)
                    if disocclusion_mask.sum() > 0:
                        flow = compute_raft_flow(models, anchor, base)
                        flow_y, flow_x = flow[..., 1], flow[..., 0]
                        grid_y, grid_x = np.mgrid[0:height, 0:width]
                        map_x = (grid_x + flow_x).astype(np.float32)
                        map_y = (grid_y + flow_y).astype(np.float32)
                        warped = cv2.remap(anchor, map_x, map_y, cv2.INTER_LINEAR)
                        
                        blend_mask = cv2.GaussianBlur(disocclusion_mask, (5, 5), 0.0)[..., None]
                        rendered = (rendered.astype(np.float32) * (1.0 - blend_mask) + warped.astype(np.float32) * blend_mask).astype(np.uint8)
                        coverage = np.clip(coverage + disocclusion_mask, 0.0, 1.0)
                        
            confidence = render_confidence(
                base,
                rendered,
                coverage,
                base_actor,
                pose_quality,
                args.photo_sigma,
                previous_confidence,
                args.confidence_ema,
            )
            for name, alpha in variants.items():
                effective_alpha = alpha
                if name == "depth_layered_hdecay":
                    if args.alpha_decay == "linear":
                        effective_alpha = alpha * (1.0 - frame_index / len(bases))
                    elif args.alpha_decay == "exponential":
                        effective_alpha = alpha * (args.alpha_decay_rate ** frame_index)
                    elif args.horizon_decay and frame_index > args.horizon_decay_start:
                        span = max(1, args.horizon_decay_end - args.horizon_decay_start)
                        progress = min(1.0, (frame_index - args.horizon_decay_start) / span)
                        effective_alpha = alpha + progress * (args.horizon_decay_floor - alpha)
                        
                outputs[name].append(compose(
                    base, rendered, confidence, effective_alpha, 
                    depth=depth, use_depth_weighting=args.alpha_depth_weighting
                ))
            previous_confidence = confidence
            diagnostics.append(
                {
                    "sample_id": row["sample_id"],
                    "frame": frame_index,
                    **pose_diag,
                    "pose_quality": pose_quality,
                    "pose_filled": filled,
                    "coverage": float(coverage.mean()),
                    "actor_fraction": float(base_actor.mean()),
                    "mean_confidence": float(confidence.mean()),
                }
            )
        fps = video_fps(cv2_module, base_video) or 30.0
        for name in variants:
            video_path = args.out_root / name / "videos" / f"{row['sample_id']}.avi"
            write_video(
                cv2_module,
                outputs[name],
                video_path,
                fps,
                prepend_guard=not args.preserve_full_video,
            )
            reports[name].append(
                {"sample_id": row["sample_id"], "video": str(video_path), "status": "ok", "future_frames": len(outputs[name])}
            )
        print(f"[{position}/{len(rows)}] {row['sample_id']} | ok", flush=True)

    for name, report_rows in reports.items():
        if not report_rows:
            continue
        path = args.out_root / name / "report.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
    if diagnostics:
        with (args.out_root / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostics[0].keys())
            writer.writeheader()
            writer.writerows(diagnostics)
    print(f"output={args.out_root}")


if __name__ == "__main__":
    main()
