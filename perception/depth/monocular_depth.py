# perception/depth/scaled_monocular_depth.py

import cv2
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
from PIL import Image as PILImage
import yaml
from pathlib import Path


class ScaledMonocularDepth:
    """
    Depth Anything V2 monocular depth estimation,
    scaled to metric units using a PnP AprilTag anchor on the gripper.

    Pipeline:
        1. Run Depth Anything V2 → relative depth map (unitless)
        2. Detect AprilTag on gripper → true metric Z from PnP tvec
        3. Sample relative depth at tag center pixel → patch median
        4. Compute scale = true_Z / relative_Z_at_tag
        5. Apply scale globally → metric depth map
        6. Back-project any pixel (u, v) → 3D point in camera frame

    Coordinate convention:
        Camera frame: OpenCV standard (+X right, +Y down, +Z forward)
        All units: metres unless stated otherwise.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: str = 'configs/config.yaml'):

        # --- Master config ---
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"Master config not found at '{config_path}'."
            )
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.max_depth = cfg['localization'].get('max_depth_meters', 5.0)

        # FIXED: Parsed calibration from centralized YAML format instead of legacy np.load()
        calib_file = cfg['camera']['calibration_file']
        if not Path(calib_file).exists():
            raise FileNotFoundError(
                f"Camera calibration file not found at '{calib_file}'. "
                "Run run_calibration.py first."
            )
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)

        self.K = np.array(
            calib_data['camera_matrix']['data'], dtype=np.float64
        ).reshape((3, 3))

        # FIXED: Distortion coefficients loaded successfully to handle wide-angle lenses
        self.D = np.array(
            calib_data['distortion_coefficients']['data'], dtype=np.float64
        ).reshape((1, -1))

        self.image_w = int(calib_data['image_width'])
        self.image_h = int(calib_data['image_height'])

        # FIXED: Configurable depth model architecture with safe small-hf fallback
        model_id = cfg.get('depth_model', {}).get(
            'model_id', 'depth-anything/Depth-Anything-V2-Small-hf'
        )

        print(f"Loading depth model: {model_id}")
        self.depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Depth model loaded successfully.")

        # FIXED: Configurable tracking patch size for robust scale sampling
        self.anchor_patch_size = cfg.get('depth_model', {}).get(
            'anchor_patch_size', 5
        )

        # FIXED: Added temporal smoothing placeholder to prevent multi-frame jitter
        self._last_valid_scale = None

    # ------------------------------------------------------------------ #
    #  Metric Depth Map                                                  #
    # ------------------------------------------------------------------ #

    def get_scaled_depth_map(self,
                             frame_bgr: np.ndarray,
                             tag_tvec,
                             tag_center_2d) -> np.ndarray | None:
        """
        Generate a full metric depth map anchored by the AprilTag.

        Args:
            frame_bgr     : BGR image from Arducam (H x W x 3, uint8)
            tag_tvec      : PnP translation vector of tag in camera frame.
                            Accepts shape (3,), (3,1), or (1,3).
            tag_center_2d : (u, v) pixel coordinates of tag centre in image.

        Returns:
            depth_metric  : float64 array (H x W) in metres, or None on failure.
        """
        # --- Run monocular depth model ---
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        result  = self.depth_pipe(pil_img)

        # FIXED: Instantiated directly as float64 to remove implicit real-time loop upcasting overhead
        depth_rel = np.array(result['depth'], dtype=np.float64)

        # Resize to match input frame if model output differs
        if depth_rel.shape[:2] != (frame_bgr.shape[0], frame_bgr.shape[1]):
            depth_rel = cv2.resize(
                depth_rel,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # FIXED: Unconditional vector flattening to guarantee safe scalar indexing from PnP shapes
        flat_tvec  = np.array(tag_tvec, dtype=np.float64).flatten()
        true_depth = float(flat_tvec[2])   # Z-forward in camera frame (metres)

        if true_depth <= 0.0:
            print("[WARNING] Tag tvec Z is non-positive — invalid PnP result.")
            return None

        # --- Sample relative depth at tag anchor pixel ---
        tx = int(np.clip(tag_center_2d[0], 0, depth_rel.shape[1] - 1))
        ty = int(np.clip(tag_center_2d[1], 0, depth_rel.shape[0] - 1))

        ps = self.anchor_patch_size
        x0 = max(0, tx - ps);  x1 = min(depth_rel.shape[1], tx + ps)
        y0 = max(0, ty - ps);  y1 = min(depth_rel.shape[0], ty + ps)

        patch = depth_rel[y0:y1, x0:x1]

        # Filter NaN / Inf values before median
        patch = patch[np.isfinite(patch)]
        if len(patch) == 0:
            print("[WARNING] Depth patch at tag anchor contains no finite values.")
            return self._apply_fallback_scale(depth_rel)

        tag_depth_rel = float(np.median(patch))

        # FIXED: Scale-independent relative metric threshold utilizing standard variance (30%)
        patch_std = float(patch.std())
        if patch_std > 0.3 * abs(tag_depth_rel):
            print(
                f"[WARNING] High depth variance at anchor patch "
                f"(std={patch_std:.4f}, median={tag_depth_rel:.4f}). "
                "Anchor point may be on a depth discontinuity."
            )
            return self._apply_fallback_scale(depth_rel)

        if tag_depth_rel < 1e-6:
            print("[WARNING] Relative depth at tag anchor is near zero.")
            return self._apply_fallback_scale(depth_rel)

        # --- Compute and apply scale ---
        scale = true_depth / tag_depth_rel
        depth_metric = depth_rel * scale

        # FIXED: Max scale constraint safety ceiling added to block explosion from occlusion anomalies
        MAX_PLAUSIBLE_SCALE = 50.0   # relative-to-metric; adjust per model
        if scale > MAX_PLAUSIBLE_SCALE or scale < 0.01:
            print(
                f"[WARNING] Computed scale factor {scale:.4f} is implausible. "
                "Rejecting — check tag visibility and PnP result."
            )
            return self._apply_fallback_scale(depth_rel)

        # Store for temporal fallback
        self._last_valid_scale = scale

        return depth_metric

    # ------------------------------------------------------------------ #
    #  Back-Projection                                                   #
    # ------------------------------------------------------------------ #

    def pixel_to_3d(self,
                    u: float,
                    v: float,
                    depth_map: np.ndarray) -> np.ndarray | None:
        """
        Back-project a pixel coordinate (u, v) to a 3D point in camera frame.

        Args:
            u, v      : pixel coordinates (float or int, will be clipped)
            depth_map : metric depth map (H x W float64) from get_scaled_depth_map

        Returns:
            [X, Y, Z] in camera frame (metres), or None if depth is invalid.
        """
        h, w = depth_map.shape

        # Clip to valid array bounds
        u_c = int(np.clip(u, 0, w - 1))
        v_c = int(np.clip(v, 0, h - 1))

        depth = float(depth_map[v_c, u_c])

        if not np.isfinite(depth) or depth <= 0.0 or depth > self.max_depth:
            return None

        fx = self.K[0, 0];  fy = self.K[1, 1]
        cx = self.K[0, 2];  cy = self.K[1, 2]

        # FIXED: Enforced consistent calculation by matching formulas with clipped variables (u_c, v_c)
        X = (u_c - cx) * depth / fx
        Y = (v_c - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z], dtype=np.float64)

    def pixel_to_3d_undistorted(self,
                                 u: float,
                                 v: float,
                                 depth_map: np.ndarray) -> np.ndarray | None:
        """
        Back-project after undistorting the pixel coordinate via self.D.
        Use this for wide-angle Arducam lenses where barrel distortion is
        significant (k1 > 0.1), especially for pixels near image edges.
        """
        h, w = depth_map.shape

        # 1. FIXED spatial mismatch: Clip and lookup target distance on the RAW coordinate layout first
        u_c = int(np.clip(u, 0, w - 1))
        v_c = int(np.clip(v, 0, h - 1))
        depth = float(depth_map[v_c, u_c])

        if not np.isfinite(depth) or depth <= 0.0 or depth > self.max_depth:
            return None

        # 2. Extract the ideal pinhole ray location using OpenCV undistort mapping parameters
        pts = np.array([[[float(u_c), float(v_c)]]], dtype=np.float64)
        pts_undist = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        u_ud = float(pts_undist[0, 0, 0])
        v_ud = float(pts_undist[0, 0, 1])

        # 3. Project out into metrics space leveraging raw depth along the corrected light projection ray
        fx = self.K[0, 0];  fy = self.K[1, 1]
        cx = self.K[0, 2];  cy = self.K[1, 2]

        X = (u_ud - cx) * depth / fx
        Y = (v_ud - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z], dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Diagnostic Overlay                                                #
    # ------------------------------------------------------------------ #

    def draw_depth_overlay(self,
                            frame_bgr: np.ndarray,
                            depth_metric: np.ndarray,
                            tag_center_2d=None) -> np.ndarray:
        """
        Render a colour-mapped depth overlay for visual debugging.
        Useful during integration testing to verify scale recovery visually.
        """
        # FIXED: Added diagnostic helper utility to prevent print loop reliance
        display    = frame_bgr.copy()
        valid_mask = np.isfinite(depth_metric) & (depth_metric > 0)

        if not valid_mask.any():
            return display

        d_vis = np.zeros_like(depth_metric, dtype=np.uint8)
        d_min = depth_metric[valid_mask].min()
        d_max = min(depth_metric[valid_mask].max(), self.max_depth)

        if d_max > d_min:
            norm = (depth_metric - d_min) / (d_max - d_min)
            norm = np.clip(norm, 0, 1)
            d_vis = (norm * 255).astype(np.uint8)

        colourmap  = cv2.applyColorMap(d_vis, cv2.COLORMAP_PLASMA)
        overlay    = cv2.addWeighted(display, 0.5, colourmap, 0.5, 0)

        # Draw anchor point
        if tag_center_2d is not None:
            tx = int(np.clip(tag_center_2d[0], 0, frame_bgr.shape[1] - 1))
            ty = int(np.clip(tag_center_2d[1], 0, frame_bgr.shape[0] - 1))
            cv2.circle(overlay, (tx, ty), 8, (0, 255, 255), 2)
            cv2.putText(overlay, "ANCHOR", (tx + 10, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return overlay

    # ------------------------------------------------------------------ #
    #  Internal Helpers                                                  #
    # ------------------------------------------------------------------ #

    def _apply_fallback_scale(self,
                              depth_rel: np.ndarray) -> np.ndarray | None:
        """
        Apply the last known valid scale when the current anchor fails.
        Returns None if no valid scale has ever been computed.
        """
        if self._last_valid_scale is not None:
            print(
                f"[INFO] Using fallback scale from previous frame: "
                f"{self._last_valid_scale:.4f}"
            )
            return depth_rel * self._last_valid_scale

        print("[ERROR] No valid scale available and anchor failed. "
              "Ensure tag is visible at startup.")
        return None


# ---------------------------------------------------------------------- #
#  Quick Smoke Test                                                      #
# ---------------------------------------------------------------------- #

if __name__ == '__main__':
    print("ScaledMonocularDepth — offline smoke test")
    print("(Requires a valid configs/config.yaml and calibration file)\n")

    try:
        smd = ScaledMonocularDepth()

        # Simulate a 640x480 grey frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate tag at 0.5 m depth, centred in image
        dummy_tvec      = np.array([[0.0], [0.0], [0.5]]) # Testing PnP structural column shape stability
        dummy_center_2d = (320, 240)

        depth_map = smd.get_scaled_depth_map(dummy_frame, dummy_tvec, dummy_center_2d)

        if depth_map is not None:
            pt = smd.pixel_to_3d_undistorted(320, 240, depth_map)
            print(f"3D point at image centre: {pt}")
        else:
            print("Depth map returned None (expected with a blank dummy frame due to zero relative variance).")
    except FileNotFoundError as e:
        print(f"Offline test skipped gracefully: {e}")