# perception/depth/scaled_depth.py
#
# Scaled monocular depth using AprilTag as metric anchor.
#
# YOUR RESPONSIBILITY:
#   Implement MonocularDepthModel in perception/depth/your_dmd_model.py
#   It must expose:
#       depth_map = model.infer(frame_bgr)
#           Returns: HxW float32 numpy array of RELATIVE depth values
#           (values are arbitrary scale, not metric)
#
# RESOURCES FOR DMD IMPLEMENTATION:
#   Depth Anything v2 (recommended — best accuracy/speed tradeoff):
#       Paper:  https://arxiv.org/abs/2406.09414
#       Code:   https://github.com/DepthAnything/Depth-Anything-V2
#       HuggingFace: depth-anything/Depth-Anything-V2-Small-hf
#                    depth-anything/Depth-Anything-V2-Base-hf
#       Use Small for Raspberry Pi, Base for Jetson
#
#   UniDepth (metric depth without anchor — alternative approach):
#       Paper:  https://arxiv.org/abs/2403.18913
#       Code:   https://github.com/lpiccinelli-eth/UniDepth
#
#   Metric3D v2 (another metric depth option):
#       Paper:  https://arxiv.org/abs/2404.15506
#       Code:   https://github.com/YvanYin/Metric3D
#
#   ZoeDepth (older but well documented):
#       Code:   https://github.com/isl-org/ZoeDepth
#
#   For keypoint extraction from depth features:
#       SuperPoint:  https://github.com/magicleap/SuperPointPretrainedNetwork
#       XFeat:       https://github.com/verlab/accelerated_features
#       (these give you geometric keypoints from RGB)
#
#   For SEMANTIC keypoints (action-relevant, not just geometric):
#       Transporter / Keypoint3D:
#           https://arxiv.org/abs/2106.06272
#       Category-level keypoints (NOCS):
#           https://arxiv.org/abs/1901.02970
#       Functional keypoints for manipulation:
#           https://arxiv.org/abs/2109.12961  (KeyPose)
#
# HOW SCALING WORKS:
#   1. PnP gives absolute depth Z_tag of gripper tag in camera frame
#   2. DMD gives relative depth map D_rel (arbitrary units)
#   3. At the tag pixel location: D_rel[tag_y, tag_x] = d_anchor
#   4. Scale factor: s = Z_tag / d_anchor
#   5. Metric depth map: D_metric = D_rel * s
#   6. Object depth: Z_object = D_metric[object_y, object_x]
#
# IMPORTANT NOTE ON ACCURACY:
#   The scale factor is computed at one point (the tag).
#   Monocular depth models are not perfectly scale-consistent across the image.
#   Accuracy is best near the tag and degrades further away.
#   For a manipulation workspace of 40-80cm, error is typically 3-8%.
#   This is sufficient for adaptive gripper grasping (+-15mm tolerance).
#
# COORDINATE FRAME NOTE:
#   pixel_to_3d() returns positions in OpenCV camera frame:
#   +X right, +Y down, +Z forward.
#   Transform through T_cam_to_base before passing to arm controller or MoveIt2.

# perception/depth/scaled_depth.py

import numpy as np
import cv2
import yaml
from pathlib import Path


class ScaledDepthLocalizer:
    """
    Converts monocular relative depth to metric depth using
    the AprilTag's known Z-depth (optical axis) as a scale anchor.

    Scale model:
        scale        = tag_tvec[2] / d_anchor   (Z-axis ONLY, not Euclidean norm)
        depth_metric = depth_rel * scale         (NO normalization before scaling)

    All outputs are in OpenCV camera frame: +X right, +Y down, +Z forward.
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Master config not found at '{config_path}'")

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        calib_file = cfg['camera']['calibration_file']
        if not Path(calib_file).exists():
            raise FileNotFoundError(
                f"Camera calibration not found at '{calib_file}'. Run run_calibration.py first."
            )
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)

        self.K  = np.array(
            calib_data['camera_matrix']['data'], dtype=np.float64
        ).reshape((3, 3))
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])

        # Depth validity range from config
        loc_cfg        = cfg.get('localization', {})
        self.min_depth = float(loc_cfg.get('min_depth_meters', 0.05))
        self.max_depth = float(loc_cfg.get('max_depth_meters', 5.0))

        # Patch sampling radius for anchor and pixel_to_3d
        self.anchor_patch_radius = int(loc_cfg.get('anchor_patch_radius', 5))
        self.lookup_patch_radius = int(loc_cfg.get('lookup_patch_radius', 3))

        # Temporal EMA state for pixel_to_3d output
        self._ema_alpha   = float(loc_cfg.get('depth_ema_alpha', 0.0))
        self._last_pos_3d = None   # last smoothed 3D position

        # Injected depth model
        self.depth_model = None

    def set_depth_model(self, model):
        """Inject the monocular depth model (MonocularDepthModel instance)."""
        self.depth_model = model

    def get_scaled_depth_map(self,
                             frame_bgr: np.ndarray,
                             tag_tvec,
                             tag_center_2d,
                             patch_radius: int = None) -> np.ndarray | None:
        """Generate a metric depth map anchored at the AprilTag Z-depth."""
        if self.depth_model is None:
            raise RuntimeError(
                "No depth model set. Call set_depth_model() before inference."
            )
        if frame_bgr is None or frame_bgr.ndim != 3:
            print("[ScaledDepthLocalizer] ERROR: frame_bgr must be HxWx3 BGR array.")
            return None

        pr = patch_radius if patch_radius is not None else self.anchor_patch_radius

        # Run monocular depth inference
        depth_rel = self.depth_model.infer(frame_bgr)
        if depth_rel is None:
            return None
        
        print("\n===== SCALING DEBUG =====")

        print(
            f"Raw depth range: "
            f"{depth_rel.min():.4f} "
            f"to "
            f"{depth_rel.max():.4f}"
        )

        

        # Accept native precision directly to save downstream allocation cycles
        depth_rel = np.asarray(depth_rel)

        if not np.any(np.isfinite(depth_rel)):
            print("[ScaledDepthLocalizer] WARNING: depth map has no finite values.")
            return None

        # Resize to match frame if needed
        h, w = frame_bgr.shape[:2]
        if depth_rel.shape[:2] != (h, w):
            # INTER_NEAREST preserves raw depth values at structural boundaries
            depth_rel = cv2.resize(
                depth_rel.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        # Extract true Z-depth from tag tvec (Z-axis anchoring)
        tvec_3d = np.array(tag_tvec, dtype=np.float64).flatten()
        z_true = float(tvec_3d[2])

        if z_true <= 0.0:
            print("[ScaledDepthLocalizer] WARNING: tag Z-depth is non-positive — invalid PnP result.")
            return None

        # Sample anchor patch at tag pixel
        tx = int(np.clip(int(np.floor(tag_center_2d[0])), pr, w - pr - 1))
        ty = int(np.clip(int(np.floor(tag_center_2d[1])), pr, h - pr - 1))

        patch        = depth_rel[ty - pr: ty + pr, tx - pr: tx + pr]
        patch_finite = patch[np.isfinite(patch)]

        if len(patch_finite) == 0:
            print("[ScaledDepthLocalizer] WARNING: no finite values in anchor patch.")
            return None

        d_anchor = float(np.median(patch_finite))

        print(
            f"DepthAnything value at tag: "
            f"{d_anchor:.6f}"
        )

        print(
            f"Tag raw value = {d_anchor:.6f}"
        )

        '''
        print(
            f"Scale factor: "
            f"{scale:.6f}"
        )
        '''

        print("=========================\n")

        # Variance check — high std means tag is on an unstable depth boundary
        if float(np.std(patch_finite)) > 0.5 * abs(d_anchor):
            print("[ScaledDepthLocalizer] WARNING: high anchor patch variance — tag may be on depth boundary.")
            return None

        # Scale explosion guard
        if d_anchor < 0.02:
            print(f"[ScaledDepthLocalizer] WARNING: d_anchor={d_anchor:.5f} too small.")
            return None

        scale = z_true / d_anchor

        # TEST: assume depth_rel behaves like inverse depth
        depth_metric = z_true * (
            d_anchor /
            np.maximum(depth_rel, 1e-6)
)

        return depth_metric.astype(np.float64)

    def pixel_to_3d(self,
                    u: float,
                    v: float,
                    depth_metric_map: np.ndarray) -> np.ndarray | None:
        """Back-project pixel (u, v) to 3D using metric depth map."""
        h, w = depth_metric_map.shape

        # Clean integer casting sequence to prevent out-of-bounds boundary errors
        u_floor = int(np.floor(u))
        v_floor = int(np.floor(v))
        u_i = int(np.clip(u_floor, 0, w - 1))
        v_i = int(np.clip(v_floor, 0, h - 1))

        r   = self.lookup_patch_radius
        u0, u1 = max(0, u_i - r), min(w, u_i + r + 1)
        v0, v1 = max(0, v_i - r), min(h, v_i + r + 1)

        patch = depth_metric_map[v0:v1, u0:u1]
        valid = patch[
            (patch > self.min_depth) &
            (patch < self.max_depth) &
            np.isfinite(patch)
        ]

        if len(valid) == 0:
            return None

        depth = float(np.median(valid))

        # OpenCV camera frame projection math using clipped coordinates
        X = (u_i - self.cx) * depth / self.fx
        Y = (v_i - self.cy) * depth / self.fy
        Z = depth

        pos_3d = np.array([X, Y, Z], dtype=np.float64)

        # EMA temporal filtering on 3D output
        if self._ema_alpha > 0.0 and self._last_pos_3d is not None:
            pos_3d = (self._ema_alpha * pos_3d + (1.0 - self._ema_alpha) * self._last_pos_3d)
        
        self._last_pos_3d = pos_3d.copy()

        return pos_3d

    def get_object_3d(self,
                      frame_bgr: np.ndarray,
                      object_centroid_2d,
                      tag_tvec,
                      tag_center_2d):
        """Single-object convenience wrapper."""
        depth_map = self.get_scaled_depth_map(frame_bgr, tag_tvec, tag_center_2d)
        if depth_map is None:
            return None, None

        pos_3d = self.pixel_to_3d(object_centroid_2d[0], object_centroid_2d[1], depth_map)
        return pos_3d, depth_map

    def get_multiple_objects_3d(self,
                                frame_bgr: np.ndarray,
                                centroids_2d: list,
                                tag_tvec,
                                tag_center_2d):
        """Multi-object wrapper — runs depth inference ONCE, reuses map."""
        depth_map = self.get_scaled_depth_map(frame_bgr, tag_tvec, tag_center_2d)
        if depth_map is None:
            return [None] * len(centroids_2d), None

        positions = [
            self.pixel_to_3d(c[0], c[1], depth_map)
            for c in centroids_2d
        ]
        return positions, depth_map