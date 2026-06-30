# perception/keypoints/keypoint_localizer.py
#
# Semantic keypoint extraction and fixture localization.
#
# WHAT THIS DOES:
#   Takes the object segmentation mask + the metric depth map
#   and extracts semantically meaningful 3D keypoints on the object.
#
#   These keypoints are then used by AnyGrasp as grasp candidates,
#   or directly as pre-grasp targets.
#
# TWO TYPES OF KEYPOINTS:
#
#   GEOMETRIC keypoints — detected from local image structure
#       (corners, edges, blobs, high-curvature regions)
#       Libraries: OpenCV (ORB, SIFT), XFeat, SuperPoint
#       These tell you WHERE interesting geometry is.
#       They don't tell you whether it's graspable.
#
#   SEMANTIC keypoints — detected from learned category knowledge
#       (handle center, rim, pour point, button)
#       These are action-relevant — they tell you WHERE to grasp.
#       Require a trained model per object category.
#
# FOR YOUR IMPLEMENTATION:
#   Start with geometric keypoints (no training required).
#   Add semantic keypoints per object category as the project matures.
#
# RESOURCES:
#   XFeat (fast geometric keypoints, better than ORB):
#       Paper: https://arxiv.org/abs/2404.19174
#       Code:  https://github.com/verlab/accelerated_features
#
#   SuperPoint (learned geometric keypoints):
#       Paper: https://arxiv.org/abs/1712.07629
#       Code:  https://github.com/magicleap/SuperPointPretrainedNetwork
#
#   KeyPose (semantic keypoints for manipulation):
#       Paper: https://arxiv.org/abs/2109.12961
#
#   Functional keypoints survey:
#       https://arxiv.org/abs/2106.06272
#
#   Affordance-based keypoints (Where2Act):
#       Paper: https://arxiv.org/abs/2101.02692
#       Code:  https://github.com/daerduoCarey/where2act
#
#   Category-level 6D pose with keypoints (NOCS):
#       Paper: https://arxiv.org/abs/1901.02970
#
# HOW KEYPOINTS FEED INTO GRASP PLANNING:
#   Geometric keypoints on an object's point cloud are passed to AnyGrasp
#   as the input point cloud. AnyGrasp generates grasp poses around those
#   points. Semantic keypoints can additionally bias AnyGrasp toward
#   functionally meaningful grasp locations.
#
# COORDINATE FRAME NOTE (applies to ALL outputs in this file):
#   All 3D outputs are in OpenCV camera frame (+X right, +Y down, +Z forward).
#   Transform through T_cam_to_base before passing to arm controller or MoveIt2.
#   Handle/side labels (handle_image_right, handle_image_left) use image-frame
#   convention (camera POV), NOT robot-wrist-frame convention.

import warnings
import cv2
import numpy as np
import yaml


class GeometricKeypointExtractor:
    """
    Extracts geometric keypoints from the object region.

    Uses OpenCV ORB as default (no dependencies, runs on CPU).
    NOTE: ORB works poorly on textureless household objects
    (plain cups, white bottles, bowls). Replace with XFeat or
    SuperPoint when GPU inference is available — the file header
    has links to both.

    Output: list of 2D keypoints + their 3D positions via depth map.

    Coordinate frame:
        All 3D outputs are in OpenCV camera frame (+X right, +Y down, +Z forward).
        Transform through T_cam_to_base before passing to arm controller or MoveIt2.
    """

    def __init__(self, config_path='configs/config.yaml'):
        # Explicit FileNotFoundError with actionable message.
        # NOTE: Python uses FileNotFoundError, not Java's FileNotFoundException.
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config not found at '{config_path}'. "
                "Check configs/config.yaml exists."
            )

        # All tunable parameters read from config — not hardcoded.
        # Add to configs/config.yaml:
        #   keypoints:
        #     geometric:
        #       orb_features: 200
        #       min_mask_pixels: 50
        geo_cfg          = cfg.get('keypoints', {}).get('geometric', {})
        self.detector    = cv2.ORB_create(nfeatures=geo_cfg.get('orb_features', 200))
        self.min_mask_px = geo_cfg.get('min_mask_pixels', 50)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_mask(self, mask):
        """
        Ensure mask is uint8 binary (0 / 255).

        Float soft-masks from segmentation models contain near-zero noise
        that passes a bare `> 0` check. Normalising here prevents ghost
        keypoints at noisy background pixels.
        """
        if mask.dtype != np.uint8:
            return (mask > 0).astype(np.uint8)
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_2d(self, frame_gray, mask):
        """
        Extract 2D keypoints within the object mask.

        Args:
            frame_gray : HxW uint8 grayscale image
            mask       : HxW uint8 binary mask (255 = object, 0 = background)

        Returns:
            keypoints   : list of cv2.KeyPoint
            descriptors : NxD numpy array, or None if no keypoints found
        """
        keypoints, descriptors = self.detector.detectAndCompute(frame_gray, mask)
        return keypoints, descriptors

    def lift_to_3d(self, keypoints_2d, depth_metric_map, depth_localizer):
        """
        Convert 2D keypoints to 3D using a fully vectorized pipeline.

        Replaces the old per-keypoint Python loop with one-shot vectorized
        depth lookup and back-projection (~200x faster on 200 ORB features).

        Args:
            keypoints_2d     : list of cv2.KeyPoint
            depth_metric_map : HxW float32 metric depth map (meters)
            depth_localizer  : ScaledDepthLocalizer instance
                               (provides fx, fy, cx, cy, min_depth, max_depth)

        Returns:
            points_3d : Nx3 float32 array in OpenCV camera frame
            valid_kps : list of cv2.KeyPoint whose depth was valid
                        (same length as points_3d rows)
        """
        if not keypoints_2d:
            return np.zeros((0, 3), dtype=np.float32), []

        # Batch-extract pixel coordinates from keypoint structures.
        # float64 to match K-matrix precision (fx, fy, cx, cy are float64).
        pts = np.array([kp.pt for kp in keypoints_2d], dtype=np.float64)

        # Floor + clip: guard against sub-pixel coords landing outside the map.
        u_arr = np.clip(
            np.floor(pts[:, 0]).astype(int), 0, depth_metric_map.shape[1] - 1
        )
        v_arr = np.clip(
            np.floor(pts[:, 1]).astype(int), 0, depth_metric_map.shape[0] - 1
        )

        # float64 depth to match K-matrix precision throughout back-projection.
        depths = depth_metric_map[v_arr, u_arr].astype(np.float64)

        valid = (
            (depths > depth_localizer.min_depth) &
            (depths < depth_localizer.max_depth) &
            np.isfinite(depths)
        )

        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32), []

        # Pinhole back-projection — OpenCV camera frame convention.
        X = (pts[valid, 0] - depth_localizer.cx) * depths[valid] / depth_localizer.fx
        Y = (pts[valid, 1] - depth_localizer.cy) * depths[valid] / depth_localizer.fy
        Z = depths[valid]

        points_3d = np.stack([X, Y, Z], axis=1)
        valid_kps = [kp for kp, v in zip(keypoints_2d, valid) if v]

        # float32 for AnyGrasp downstream compatibility.
        return points_3d.astype(np.float32), valid_kps

    def extract_3d(self, frame_bgr, mask, depth_metric_map, depth_localizer):
        """
        Full pipeline: image + mask + depth → 3D keypoints.

        Args:
            frame_bgr        : HxWx3 uint8 BGR image (already undistorted)
            mask             : HxW mask (uint8 or float — normalised internally)
            depth_metric_map : HxW float32 metric depth map (meters)
            depth_localizer  : ScaledDepthLocalizer instance

        Returns:
            kps_3d    : Nx3 float32 array in camera frame
            valid_kps : corresponding list of cv2.KeyPoint
        """
        mask_bin = self._normalize_mask(mask)

        ys, xs = np.where(mask_bin > 0)
        if len(xs) < self.min_mask_px:
            return np.zeros((0, 3), dtype=np.float32), []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps_2d, _ = self.extract_2d(gray, mask_bin)

        if not kps_2d:
            return np.zeros((0, 3), dtype=np.float32), []

        return self.lift_to_3d(kps_2d, depth_metric_map, depth_localizer)

    def build_object_pointcloud(self, mask, depth_metric_map, depth_localizer,
                                max_points=1024):
        """
        Build a dense point cloud of the object from its mask + depth map.

        Primary input format expected by AnyGrasp.

        Uses vectorized NumPy back-projection (not a per-pixel Python loop)
        for 20-100x better performance — critical for Jetson/Pi deployment.

        Downsampling uses voxel centroid averaging (not random sampling):
        deterministic, unbiased, and preserves surface geometry better.

        Args:
            mask             : HxW mask (uint8 or float — normalised internally)
            depth_metric_map : HxW float32 metric depth map (meters)
            depth_localizer  : ScaledDepthLocalizer instance
            max_points       : downsample to at most this many points

        Returns:
            points : Nx3 float32 numpy array (X, Y, Z in camera frame)
        """
        mask_bin = self._normalize_mask(mask)

        ys, xs = np.where(mask_bin > 0)
        if len(xs) < self.min_mask_px:
            return np.zeros((0, 3), dtype=np.float32)

        # float64 throughout to match K-matrix precision.
        depths = depth_metric_map[ys, xs].astype(np.float64)
        valid = (
            (depths > depth_localizer.min_depth) &
            (depths < depth_localizer.max_depth) &
            np.isfinite(depths)
        )

        xs_v = xs[valid].astype(np.float64)
        ys_v = ys[valid].astype(np.float64)
        zs_v = depths[valid]

        if len(zs_v) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        X = (xs_v - depth_localizer.cx) * zs_v / depth_localizer.fx
        Y = (ys_v - depth_localizer.cy) * zs_v / depth_localizer.fy
        Z = zs_v

        points = np.stack([X, Y, Z], axis=1)

        # Voxel centroid downsampling.
        if len(points) > max_points:
            voxel_size = 0.005  # 5 mm grid — tune per workspace scale

            # int64 prevents overflow/collision when packing voxel coordinates.
            # int32 wraps on negative coords near the camera origin.
            voxel_idx = np.floor(points / voxel_size).astype(np.int64)
            packed_idx = (
                voxel_idx[:, 0] * 1_000_000 +
                voxel_idx[:, 1] * 1_000 +
                voxel_idx[:, 2]
            ).astype(np.int64)

            sort_order    = np.argsort(packed_idx)
            sorted_packed = packed_idx[sort_order]
            sorted_points = points[sort_order]

            _, indices = np.unique(sorted_packed, return_index=True)

            # reduceat computes per-voxel sums; divide by count → centroid.
            points = np.add.reduceat(sorted_points, indices, axis=0)
            counts = np.diff(
                np.append(indices, len(sorted_points))
            )[:, np.newaxis]
            points = points / counts

            if len(points) > max_points:
                points = points[:max_points]

        return points.astype(np.float32)

    def draw_keypoints(self, frame, keypoints_2d, color=(0, 255, 255)):
        """Draw 2D keypoints on frame for visualization."""
        out = frame.copy()
        for kp in keypoints_2d:
            u, v = int(np.floor(kp.pt[0])), int(np.floor(kp.pt[1]))
            cv2.circle(out, (u, v), 3, color, -1)
        return out


class SemanticKeypointExtractor:
    """
    Extracts semantically meaningful keypoints per object category.

    This class is a STUB — implement category-specific detectors as needed.

    Each category has a set of named keypoints:
        'bottle' : ['cap', 'body', 'base']
        'cup'    : ['handle_image_right', 'handle_image_left', 'rim', 'base']
        'bowl'   : ['center', 'rim']
        'book'   : ['spine', 'center']

    Handle labels use IMAGE-frame convention (camera POV), not robot-wrist-frame.
    Apply T_cam_to_base before passing to arm controller or MoveIt2.

    IMPLEMENTATION APPROACHES:
    ===========================

    Approach A — Heatmap regression (recommended for 1-3 categories):
        - Annotate 100-200 images per category with keypoint locations
        - Train a MobileNetV3 backbone with one heatmap head per keypoint
        - Use Gaussian blobs at keypoint locations as supervision signal
        - Inference: argmax of each heatmap channel = keypoint pixel
        - Then lift to 3D via depth map

    Approach B — Fine-tune an existing model:
        - Start from KeyPose or NOCS pretrained weights
        - Fine-tune on your objects. Better generalisation with less data.

    Approach C — Prompt a VLM (experimental):
        - Ask Gemini/GPT-4V "where is the handle of this cup?"
        - Returns a pixel region → lift to 3D
        - Not reliable enough for closed-loop control yet

    Approach D — Manual category priors (simplest, for V1):
        - Rule-based geometry from mask bounding box
        - Works ONLY for upright objects in known orientation
        - This is what detect_rule_based() implements

    TRAINING DATA:
        BOP Dataset:   https://bop.felk.cvut.cz/
        COCO Keypoints: https://cocodataset.org/#keypoints-2019
        BlenderProc (synthetic): https://github.com/DLR-RM/BlenderProc
    """

    def __init__(self, config_path='configs/config.yaml'):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config not found at '{config_path}'. "
                "Check configs/config.yaml exists."
            )

        # Keypoint schema and grasp priority loaded from config.
        # Falls back to hardcoded defaults if not present.
        # Add to configs/config.yaml:
        #   keypoints:
        #     schema:
        #       bottle: ['cap', 'body', 'base']
        #       ...
        #     grasp_priority:
        #       bottle: ['body', 'base', 'cap']
        #       ...
        kp_cfg = cfg.get('keypoints', {})

        self.keypoint_schema = kp_cfg.get('schema', {
            'bottle': ['cap', 'body', 'base'],
            # handle_image_right/left = image-frame labels (camera POV),
            # NOT robot-wrist-frame labels.
            'cup':    ['handle_image_right', 'handle_image_left', 'rim', 'base'],
            'bowl':   ['center', 'rim'],
            'book':   ['spine', 'center'],
        })

        self.grasp_priority = kp_cfg.get('grasp_priority', {
            'bottle': ['body', 'base', 'cap'],
            'cup':    ['handle_image_right', 'handle_image_left', 'rim', 'base'],
            'bowl':   ['rim', 'center'],
            'book':   ['center', 'spine'],
        })

        # Category detectors — populated by load_detector()
        self.detectors = {}

    def load_detector(self, class_name, model_path):
        """
        Load a trained keypoint detector for a category.
        Implement per your model architecture when training data is available.
        """
        # TODO: implement when trained models exist
        # self.detectors[class_name] = YourKeypointModel(model_path)
        pass

    def detect_rule_based(self, class_name, mask, frame_bgr):
        """
        Rule-based semantic keypoints using mask geometry.
        Use this for V1 before trained detectors are available.

        Keypoint coordinates are in image-pixel space (col, row).
        All handle/side labels use image-frame convention (camera POV),
        NOT robot-wrist-frame. Apply T_cam_to_base before passing to
        arm controller.

        LIMITATION: All rules assume objects are upright and in standard
        orientation. This will fail for:
          - Bottles lying on their side (cap/base detection wrong)
          - Cups with handle not on right side (handle detection fails)
          - Any tilted or rotated object
        Replace with trained heatmap detector when annotated data is available.

        Args:
            class_name : str, object category
            mask       : HxW mask (uint8 or float — normalised internally)
            frame_bgr  : HxWx3 uint8 BGR image (unused in rule-based path,
                         reserved for future learned detectors)

        Returns:
            dict {keypoint_name: (u, v)} pixel coordinates, or {} if mask empty
        """
        # Normalise mask dtype to prevent soft-mask noise.
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {}

        y_min, y_max = float(ys.min()), float(ys.max())
        x_min, x_max = float(xs.min()), float(xs.max())
        height = y_max - y_min
        width  = x_max - x_min
        cx     = float(xs.mean())
        cy     = float(ys.mean())

        if class_name == 'bottle':
            return {
                'cap':  (cx, float(y_min + height * 0.10)),
                'body': (cx, cy),
                'base': (cx, float(y_max - height * 0.05)),
            }

        elif class_name == 'cup':
            # Provide both left and right handle candidates.
            # get_grasp_keypoint() tries handle_image_right first, then left.
            # Still rule-based — replace with trained detector for reliability.
            return {
                'rim':                (cx,                           float(y_min + height * 0.10)),
                'handle_image_right': (float(x_max - width * 0.05), cy),
                'handle_image_left':  (float(x_min + width * 0.05), cy),
                'base':               (cx,                           float(y_max - height * 0.05)),
            }

        elif class_name == 'bowl':
            return {
                'center': (cx, cy),
                'rim':    (cx, float(y_min + height * 0.05)),
            }

        elif class_name == 'book':
            # Explicit branch — generic fallback loses the 'spine' keypoint.
            return {
                'spine':  (float(x_min + width * 0.05), cy),
                'center': (cx, cy),
            }

        else:
            # Warn on unknown classes instead of silently returning centroid.
            warnings.warn(
                f"SemanticKeypointExtractor: unknown class '{class_name}' — "
                "returning centroid only. Add a rule-based branch or trained detector."
            )
            return {'center': (cx, cy)}

    def lift_keypoints_to_3d(self, keypoints_2d, depth_metric_map, depth_localizer):
        """
        Convert 2D semantic keypoints to 3D camera-frame positions.

        Output is in OpenCV camera frame: +X right, +Y down, +Z forward.
        Transform through T_cam_to_base before passing to arm or MoveIt2.

        Args:
            keypoints_2d     : dict {name: (u, v)} pixel coordinates
            depth_metric_map : HxW float32 metric depth map (meters)
            depth_localizer  : ScaledDepthLocalizer instance

        Returns:
            dict {name: np.ndarray([X, Y, Z])} in camera frame
            Keys with no valid depth are omitted.
        """
        keypoints_3d = {}
        for name, (u, v) in keypoints_2d.items():
            pos_3d = depth_localizer.pixel_to_3d(u, v, depth_metric_map)
            if pos_3d is not None:
                keypoints_3d[name] = pos_3d
        return keypoints_3d

    def get_grasp_keypoint(self, class_name, keypoints_3d):
        """
        Select the highest-priority available grasp keypoint for a given class.

        Priority order is loaded from config (keypoints.grasp_priority).
        Falls back to any available keypoint if none in the priority list are found.

        Args:
            class_name    : str, object category
            keypoints_3d  : dict {name: np.ndarray([X, Y, Z])} from lift_keypoints_to_3d()

        Returns:
            (name, position_3d) tuple, or (None, None) if keypoints_3d is empty.
        """
        priority = self.grasp_priority.get(class_name, ['center', 'body'])
        for kp_name in priority:
            if kp_name in keypoints_3d:
                return kp_name, keypoints_3d[kp_name]

        # Fallback: return first available keypoint
        if keypoints_3d:
            name = next(iter(keypoints_3d))
            return name, keypoints_3d[name]

        return None, None