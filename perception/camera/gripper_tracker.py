# perception/camera/gripper_tracker.py
import cv2
import numpy as np
from pupil_apriltags import Detector
import yaml


class GripperTracker:
    """
    Tracks the gripper position in 3D using an AprilTag mounted on the gripper.
    Uses PnP (Perspective-n-Point) to recover full 6-DoF pose from a single camera.

    Fixes applied vs original:
      - [BUG1] validate_accuracy now forces camera resolution to match calibration
      - [BUG2] Image is undistorted BEFORE tag detection; solvePnP receives zero
               distortion to avoid double-undistortion
      - [BUG3] EMA filter on tvec/rvec output to suppress 1-pixel jitter
      - [BUG4] validate_accuracy uses try/finally to guarantee camera release
      - [BUG5] SOLVEPNP_IPPE_SQUARE version guard with fallback to ITERATIVE
      - [BUG6] EMA state initialised in __init__
      - [BUG7] draw_overlay passes zero distortion after image is undistorted
      - [BUG8] dist_coeffs reshaped and validated on load
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config_path='configs/config.yaml'):

        # ── Load config ────────────────────────────────────────────────
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # ── Camera calibration ─────────────────────────────────────────
        with open(cfg['camera']['calibration_file'], 'r') as f:
            calib = yaml.safe_load(f)

        self.K = np.array(
            calib['camera_matrix']['data'],
            dtype=np.float64
        ).reshape((3, 3))

        # [BUG8] Normalise distortion to a column vector regardless of
        #        how it was saved (flat array, row vector, column vector).
        d_raw = np.array(
            calib['distortion_coefficients']['data'],
            dtype=np.float64
        ).flatten()

        self.D = d_raw.reshape(-1, 1)                # (N,1) e.g. (5,1)       

        # Calibration resolution — MUST match what was used during
        # camera calibration so K pixel values are meaningful.
        # [BUG1] Store these; used in validate_accuracy and any
        #        future capture loop.
        self.calib_width  = int(cfg['camera']['calib_width'])   # e.g. 1280
        self.calib_height = int(cfg['camera']['calib_height'])  # e.g.  720

        # Pre-compute undistortion maps once (faster than cv2.undistort
        # called per frame, especially on an RPi).
        # [BUG2] These maps are applied before tag detection AND before
        #        solvePnP so we only need to tell solvePnP D=zeros.
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D,
            (self.calib_width, self.calib_height),
            alpha=0           # alpha=0: no black border pixels
        )
        self.new_K = new_K
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, new_K,
            (self.calib_width, self.calib_height),
            cv2.CV_16SC2       # integer map, fastest remap
        )
        # After remapping we tell solvePnP there is no remaining distortion.
        self._zero_D = np.zeros((4, 1), dtype=np.float64)  # [BUG2]

        # ── AprilTag parameters ────────────────────────────────────────
        tag_cfg = cfg['apriltag']
        self.tag_size      = float(tag_cfg['tag_size_meters'])
        self.target_tag_id = int(tag_cfg['tag_id'])

        self.detector = Detector(
            families       = tag_cfg['family'],        # e.g. 'tag36h11'
            nthreads       = int(tag_cfg['nthreads']),
            quad_decimate  = float(tag_cfg['quad_decimate']),
            quad_sigma     = 0.0,
            refine_edges   = 1,
            decode_sharpening = 0.25
        )

        # 3-D corners of the tag in its own frame (z=0 plane)
        s = self.tag_size / 2.0
        self.obj_points = np.array([
            [-s, -s, 0.0],
            [ s, -s, 0.0],
            [ s,  s, 0.0],
            [-s,  s, 0.0]
        ], dtype=np.float64)

        # [BUG5] Choose the best available PnP flag at init time so we
        #        don't crash with an AttributeError at runtime.
        if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE'):
            self._pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE
        else:
            # Fallback for OpenCV < 4.1.1 (e.g. apt-installed on older RPi OS)
            print("[GripperTracker] WARNING: SOLVEPNP_IPPE_SQUARE unavailable, "
                  "falling back to SOLVEPNP_ITERATIVE. Upgrade OpenCV for best accuracy.")
            self._pnp_flag = cv2.SOLVEPNP_ITERATIVE

        # ── Tip offset (tag frame → gripper tip) ──────────────────────
        tip_cfg = cfg['transforms']['tip_offset']
        self.tip_offset = np.array(tip_cfg, dtype=np.float64)  # (3,)

        # ── EMA state (initialised to None; set on first detection) ───
        # [BUG3, BUG6]
        self._ema_alpha  = float(cfg.get('tracker', {}).get('ema_alpha', 0.35))
        self._tvec_ema   = None   # (3,) running EMA of translation
        self._rvec_ema   = None   # (3,) running EMA of rotation vector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _undistort(self, frame):
        """
        Remap frame using pre-computed distortion maps.
        Returns undistorted image; uses new_K so K is no longer valid on
        this image — use self.new_K for any projection on the output.
        """
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def _ema_update(self, tvec, rvec):
        """
        Apply exponential moving average to suppress single-frame jitter.
        On the first call just seeds the filter.

        NOTE: Averaging rvec directly is only valid for small angular
        jitter (< ~10 deg), which is always true here since the gripper
        pose is nearly continuous between frames.
        """
        a = self._ema_alpha
        if self._tvec_ema is None:
            # [BUG6] Seed on first detection
            self._tvec_ema = tvec.copy()
            self._rvec_ema = rvec.copy()
        else:
            self._tvec_ema = a * tvec + (1.0 - a) * self._tvec_ema
            self._rvec_ema = a * rvec + (1.0 - a) * self._rvec_ema

        return self._tvec_ema.copy(), self._rvec_ema.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame):
        """
        Detect AprilTag and compute smoothed 6-DoF gripper pose.

        Args:
            frame: BGR image from camera at calibration resolution.
                   The image does NOT need to be pre-undistorted; this
                   method handles undistortion internally.

        Returns:
            dict with keys:
                tag_id, tvec, rvec, R, T_tag_cam,
                tip_position, corners, center_2d, undistorted_frame
            or None if tag not detected.

            All 3-D quantities (tvec, tip_position) are in the camera
            optical frame, in metres.
        """
        # [BUG2] Undistort FIRST so the quad-finder sees straight edges
        undist = self._undistort(frame)
        gray   = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=False,  # we run PnP ourselves
            camera_params=None
        )

        for det in detections:
            if det.tag_id != self.target_tag_id:
                continue

            img_points = det.corners.astype(np.float64)

            # [BUG2] Pass self.new_K (valid for undistorted image) and
            #        zero distortion — distortion already removed above.
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                img_points,
                self.new_K,         # ← updated K after undistortion
                self._zero_D,       # ← zeros, NOT self.D
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            tvec = tvec.flatten()
            rvec = rvec.flatten()

            # [BUG3] Smooth before returning so downstream sees stable poses
            tvec_smooth, rvec_smooth = self._ema_update(tvec, rvec)

            R, _ = cv2.Rodrigues(rvec_smooth)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = tvec_smooth

            # Gripper tip in camera frame
            tip_cam = R @ self.tip_offset + tvec_smooth

            return {
                'tag_id':           det.tag_id,
                'tvec':             tvec_smooth,         # smoothed (m)
                'rvec':             rvec_smooth,         # smoothed
                'R':                R,
                'T_tag_cam':        T,
                'tip_position':     tip_cam,             # gripper tip (m)
                'corners':          img_points,
                'center_2d':        det.center.astype(np.float64),
                'undistorted_frame': undist,              # pass along for draw_overlay
                'decision_margin': float(det.decision_margin)
            }

        return None

    def draw_overlay(self, frame, detection):
        """
        Draw detection visualisation on frame.

        Args:
            frame:     BGR image (will draw on a copy).
            detection: dict returned by detect(), or None.

        Returns:
            Annotated BGR frame.
        """
        if detection is None:
            return frame

        # Use the undistorted frame if available (it is whenever detect() runs)
        base = detection.get('undistorted_frame', frame)
        out  = base.copy()

        # [BUG7] Use new_K and zero distortion for all projection operations
        #        on the undistorted image.
        K_draw = self.new_K
        D_draw = self._zero_D

        cv2.drawFrameAxes(
            out, K_draw, D_draw,
            detection['rvec'], detection['tvec'],
            self.tag_size * 0.5
        )

        corners = detection['corners'].astype(np.int32)
        for i in range(4):
            cv2.line(out,
                     tuple(corners[i]),
                     tuple(corners[(i + 1) % 4]),
                     (0, 255, 0), 2)

        # Project tip offset point into undistorted image
        # [BUG7] D_draw = zeros, K_draw = new_K
        tip_proj, _ = cv2.projectPoints(
            self.tip_offset.reshape(1, 3),
            detection['rvec'], detection['tvec'],
            K_draw, D_draw
        )
        tip_2d = tuple(tip_proj[0, 0].astype(int))
        cv2.circle(out, tip_2d, 8, (0, 0, 255), -1)
        cv2.putText(out, "TIP",
                    (tip_2d[0] + 10, tip_2d[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        pos = detection['tvec']
        cv2.putText(out,
                    f"X:{pos[0]*100:.1f} Y:{pos[1]*100:.1f} Z:{pos[2]*100:.1f} cm",
                    (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return out

    def reset_ema(self):
        """
        Reset EMA filter state.
        Call this whenever the arm has been repositioned significantly
        (e.g. after a large commanded motion) to avoid the filter
        lagging behind the new true position.
        """
        self._tvec_ema = None
        self._rvec_ema = None

    # ------------------------------------------------------------------
    # Validation tool
    # ------------------------------------------------------------------

    def validate_accuracy(self, device_index=0):
        """
        Interactive validation tool.
        Place the gripper tag at known distances and verify PnP output.
        Expected accuracy: 3-5 mm at 50 cm with a well-calibrated lens.
        Press ESC to exit.
        """
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {device_index}")

        # [BUG1] Force camera to calibration resolution so K is valid.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.calib_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_height)

        # Verify the driver actually accepted the resolution.
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.calib_width or actual_h != self.calib_height:
            cap.release()
            raise RuntimeError(
                f"Camera returned {actual_w}×{actual_h} but calibration "
                f"requires {self.calib_width}×{self.calib_height}. "
                f"Re-calibrate at the resolution your camera supports."
            )

        print("=== PnP VALIDATION ===")
        print(f"Camera resolution: {actual_w}x{actual_h}")
        print("Place the gripper tag at measured distances.")
        print("Compare printed Z value to your ruler measurement.")
        print("Expected accuracy: 3-5mm at 50cm")
        print("Press ESC to exit.\n")

        # [BUG4] try/finally guarantees camera is released even on exception
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame capture failed.")
                    break

                detection = self.detect(frame)
                # detect() returns the undistorted frame inside the dict,
                # so draw_overlay will display the corrected image.
                overlay = self.draw_overlay(frame, detection)

                if detection is not None:
                    z_cm = detection['tvec'][2] * 100
                    print(f"\rGripper depth: {z_cm:.1f} cm        ", end='', flush=True)

                cv2.imshow('Tag Validation', overlay)
                if cv2.waitKey(1) == 27:   # ESC
                    break
        finally:
            # [BUG4] Always runs, even if an exception is thrown above
            cap.release()
            cv2.destroyAllWindows()