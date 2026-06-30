# main.py
#
# Top-level pipeline integrating all stages.
#
# PIPELINE STAGES (in order):
#   1. Camera   — Arducam, raw frame (NOT undistorted — GripperTracker handles
#                 distortion internally via solvePnP with D)
#   2. Tag      — AprilTag PnP -> absolute gripper depth in camera frame
#   3. Detect   — YOLOv8 bounding boxes on RGB frame
#   4. Segment  — FastSAM pixel mask from YOLO box prompt
#   5. Depth    — MonocularDepthModel (relative) anchored by AprilTag (metric)
#   6. Localize — Back-project centroid to 3D via metric depth
#   7. Keypts   — Geometric (ORB) + semantic keypoints on object mask
#   8. Grasp    — AnyGrasp on object point cloud -> 6-DoF grasp poses
#   9. Servo    — Visual servo arm to pre-grasp, then to grasp pose
#  10. Execute  — Close gripper, verify, lift
#  11. Retry    — On failure, offset and retry up to 3 times
#
# WHAT YOU IMPLEMENT:
#   perception/depth/your_dmd_model.py
#       class MonocularDepthModel:
#           def infer(self, frame_bgr) -> np.ndarray (HxW float32 relative depth [0,1])
#
# WHAT'S PROVIDED:
#   Everything else in this repo.

import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Top-level import — cheaper than per-call import inside run_single_grasp().
from scipy.spatial.transform import Rotation

from perception.camera.gripper_tracker import GripperTracker
from perception.detection.object_detector import ObjectDetector
from perception.depth.scaled_depth import ScaledDepthLocalizer
from perception.keypoints.keypoint_localizer import (
    GeometricKeypointExtractor,
    SemanticKeypointExtractor,
)
from planning.grasp.anygrasp_planner import AnyGraspPlanner
from planning.servo.visual_servo import VisualServoController
from planning.grasp.grasp_verifier import GraspVerifier, RetryController
from planning.motion.motion_planner import MotionPlanner
from control.arm.arm_controller import ArmController


class WheelchairArmPipeline:

    def __init__(self, config_path='configs/config.yaml'):
        print("=== Wheelchair Arm Pipeline Initializing ===\n")

        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Raises on missing required calibration; warns on optional.
        # Also initialises self.tpl (TablePlaneLocalizer) if configured.
        self._check_calibration_files()

        # ── Camera ────────────────────────────────────────────────────
        cam_cfg   = self.cfg['camera']
        self.cap  = cv2.VideoCapture(cam_cfg['device_index'])

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {cam_cfg['device_index']}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
        self.cap.set(cv2.CAP_PROP_FPS,          cam_cfg['fps'])

        # Verify the driver actually accepted the requested resolution.
        # Some Arducam/V4L2 drivers silently use a different resolution.
        # If it differs from the calibrated resolution, K is wrong and
        # every depth estimate will be off.
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != cam_cfg['width'] or actual_h != cam_cfg['height']:
            if not cam_cfg.get('allow_resolution_mismatch', False):
                raise RuntimeError(
                    f"Camera resolution mismatch: requested "
                    f"{cam_cfg['width']}x{cam_cfg['height']} but got "
                    f"{actual_w}x{actual_h}. "
                    f"K matrix is calibrated for the requested resolution — "
                    f"all depth estimates will be wrong at the actual resolution. "
                    f"Fix: recalibrate at {actual_w}x{actual_h} or set "
                    f"'camera.allow_resolution_mismatch: true' in config to suppress."
                )
            else:
                print(f"WARNING: camera resolution mismatch "
                      f"({actual_w}x{actual_h} vs calibrated "
                      f"{cam_cfg['width']}x{cam_cfg['height']}) — "
                      f"acknowledged via config.")

        # Warm up camera — discard first 30 frames while auto-exposure
        # and white balance settle (USB/Arducam specific).
        print("Warming up camera...")
        for _ in range(30):
            self.cap.read()

        # Load calibration from YAML.
        # dist_coeffs shape from yaml can be flat list → reshape to (N,1)
        # so cv2 functions always get the right shape.
        calib_file = cam_cfg['calibration_file']
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        self.K = np.array(
            calib_data['camera_matrix']['data'], dtype=np.float64
        ).reshape((3, 3))
        d_raw  = np.array(
            calib_data['dist_coeffs']['data'], dtype=np.float64
        )
        self.D = d_raw.reshape(-1, 1)

        # Tag detection cache for brief occlusion/dropout recovery.
        # If tag detection fails but cache is fresh (within TTL), use cached result.
        self.last_tag_detection = None
        self.last_tag_time      = 0.0
        self.tag_cache_ttl      = self.cfg.get('apriltag', {}).get(
            'cache_ttl_seconds', 2.0
        )

        # ── Perception subsystems ─────────────────────────────────────
        print("Loading perception subsystems...")
        self.gripper_tracker = GripperTracker(config_path)
        self.detector        = ObjectDetector(config_path)
        self.depth_localizer = ScaledDepthLocalizer(config_path)
        self.geo_keypoints   = GeometricKeypointExtractor(config_path)
        self.sem_keypoints   = SemanticKeypointExtractor(config_path)

        # Inject your monocular depth model.
        # Implement MonocularDepthModel in perception/depth/your_dmd_model.py
        self._inject_depth_model()

        # ── Planning subsystems ───────────────────────────────────────
        print("Loading planning subsystems...")
        self.grasp_planner  = AnyGraspPlanner(config_path)
        self.motion_planner = MotionPlanner(
            backend     = self.cfg['motion_planning']['backend'],
            urdf_path   = self.cfg['motion_planning']['urdf_path'],
            config_path = config_path,
        )

        # ── Control subsystems ────────────────────────────────────────
        print("Loading control subsystems...")
        self.arm      = ArmController(config_path)
        self.servo    = VisualServoController(self.arm, config_path)
        self.verifier = GraspVerifier(self.arm)
        self.retry    = RetryController(max_retries=3)

        print("\nPipeline ready.\n")

    # ─────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────

    def _check_calibration_files(self):
        """
        Raise on missing required files, warn on optional.

        - camera calibration YAML: REQUIRED — pipeline cannot start without it.
        - T_cam_to_base.npy: optional — pipeline can run in diagnostic/dryrun
          mode but grasp execution will produce wrong base-frame poses.

        Also initialises self.tpl (TablePlaneLocalizer) once here if
        use_table_plane is True, so run_perception() can use it without
        per-frame allocation.
        """
        cam_calib = self.cfg['camera']['calibration_file']
        t_cam     = self.cfg['transforms']['T_cam_to_base_file']

        if not Path(cam_calib).exists():
            raise RuntimeError(
                f"Required calibration file missing: {cam_calib}\n"
                "Run: python main.py --mode calibrate"
            )

        if not Path(t_cam).exists():
            print(f"WARNING: T_cam_to_base not found: {t_cam}")
            print("Hand-eye calibration not done. Grasp execution will use "
                  "identity transform and produce wrong base-frame poses.")
            print("Run: python main.py --mode calibrate\n")

        if self.cfg.get('localization', {}).get('use_table_plane', False):
            try:
                from perception.localization.table_plane import TablePlaneLocalizer
                self.tpl = TablePlaneLocalizer(self.config_path)
            except Exception as e:
                print(f"WARNING: Could not initialise TablePlaneLocalizer: {e}")
                self.tpl = None
        else:
            self.tpl = None

    def _inject_depth_model(self):
        """
        Load and inject your monocular depth model.

        Implement this in perception/depth/your_dmd_model.py:

            class MonocularDepthModel:
                def __init__(self):
                    pass   # load weights, initialise model

                def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
                    # frame_bgr : HxWx3 uint8 BGR
                    # returns   : HxW float32 relative depth map in [0, 1]
                    pass

        See perception/depth/your_dmd_model.py for full implementation guide.
        """
        try:
            from perception.depth.your_dmd_model import MonocularDepthModel
            model = MonocularDepthModel()
            self.depth_localizer.set_depth_model(model)
            print("Monocular depth model loaded.")
        except ImportError:
            print("WARNING: MonocularDepthModel not found.")
            print("Create perception/depth/your_dmd_model.py")
            print("Depth-based localization will be unavailable until then.")
            print("Pipeline will still run in diagnostic/detection-only mode.\n")

    # ─────────────────────────────────────────────────────────────────
    # Core frame acquisition
    # ─────────────────────────────────────────────────────────────────

    def get_frame(self):
        """
        Get raw (NOT undistorted) frame from camera.

        IMPORTANT: Do NOT undistort here. GripperTracker.detect() applies
        solvePnP with self.D on the raw frame. Undistorting here would
        double-compensate distortion. Any downstream consumer that needs
        an undistorted frame must call cv2.undistort() explicitly.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    # ─────────────────────────────────────────────────────────────────
    # Full perception stage
    # ─────────────────────────────────────────────────────────────────

    def run_perception(self, frame, target_class=None):
        """
        Run all perception stages on a single frame.

        Returns a result dict or None on failure. Dict keys:
            tag               — GripperTracker PnP result
            detection         — highest-confidence YOLO detection
            mask              — FastSAM segmentation mask (or None)
            centroid_2d       — refined centroid from mask (or YOLO center)
            angle             — principal axis angle from mask PCA (or None)
            depth_map         — metric depth map scaled by AprilTag anchor
            object_3d         — 3D object centroid in camera frame
            kps_3d            — Nx3 geometric keypoints in camera frame
            kps_2d            — corresponding 2D cv2.KeyPoint list
            sem_kps_3d        — dict {name: [X,Y,Z]} semantic keypoints
            grasp_keypoint_name — name of highest-priority grasp keypoint
            grasp_keypoint_3d   — 3D position of highest-priority grasp keypoint
            pointcloud        — Nx3 float32 object point cloud for AnyGrasp
        """
        if frame is None:
            print("PERCEPTION FAIL: frame is None")
            return None

        result  = {}
        t_stage = time.perf_counter()

        # ── Stage 2: AprilTag PnP ─────────────────────────────────────
        tag_det = self.gripper_tracker.detect(frame)

        if tag_det is None:
            age = time.time() - self.last_tag_time
            if self.last_tag_detection is not None and age < self.tag_cache_ttl:
                tag_det = self.last_tag_detection
                print(f"PERCEPTION WARN: tag not detected, using cached "
                      f"result ({age:.2f}s old)")
            else:
                print("PERCEPTION FAIL: gripper tag not detected, cache expired")
                return None
        else:
            self.last_tag_detection = tag_det
            self.last_tag_time      = time.time()

        result['tag'] = tag_det
        t0 = time.perf_counter()
        print(f"  [timing] tag detection: {(t0 - t_stage)*1000:.1f} ms")

        # ── Stage 3: Object detection ─────────────────────────────────
        detections = self.detector.detect_objects(frame)
        t1 = time.perf_counter()
        print(f"  [timing] YOLO detection: {(t1 - t0)*1000:.1f} ms")

        if not detections:
            print("PERCEPTION FAIL: no objects detected")
            return None

        if target_class:
            detections = [d for d in detections
                          if d['class_name'] == target_class]
        if not detections:
            print(f"PERCEPTION FAIL: class '{target_class}' not found")
            return None

        # Sort by confidence — ensures detections[0] is always best
        # regardless of detector output order.
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        det = detections[0]
        result['detection'] = det

        # ── Stage 4: Segmentation ─────────────────────────────────────
        mask = self.detector.segment_object(frame, det['bbox'])
        t2 = time.perf_counter()
        print(f"  [timing] FastSAM segmentation: {(t2 - t1)*1000:.1f} ms")

        centroid_2d, angle, area = self.detector.get_mask_properties(mask)

        if centroid_2d is None:
            centroid_2d = det['center_2d']
            angle       = None
            mask        = None

        result['mask']        = mask
        result['centroid_2d'] = centroid_2d
        result['angle']       = angle

        # ── Stage 5: Metric depth map (anchored by AprilTag) ──────────
        # NOTE: depth runs before keypoints — keypoints need depth_map to
        # lift 2D coords to 3D.
        depth_map = None
        if self.depth_localizer.depth_model is not None:
            depth_map = self.depth_localizer.get_scaled_depth_map(
                frame,
                tag_det['tvec'],
                tag_det['center_2d'],
            )
        t3 = time.perf_counter()
        print(f"  [timing] depth inference + scaling: {(t3 - t2)*1000:.1f} ms")

        result['depth_map'] = depth_map

        # ── Stage 6: 3D localization ──────────────────────────────────
        object_3d = None

        if depth_map is not None:
            object_3d = self.depth_localizer.pixel_to_3d(
                centroid_2d[0], centroid_2d[1], depth_map
            )

        if object_3d is None:
            # Fallback: table plane (if calibrated and configured).
            # self.tpl is initialised once at __init__ — not per frame.
            if self.tpl is not None and self.tpl.is_calibrated():
                object_3d = self.tpl.pixel_to_3d(
                    centroid_2d[0], centroid_2d[1]
                )

        if object_3d is None:
            print("PERCEPTION FAIL: could not localize object in 3D")
            return None

        result['object_3d'] = object_3d

        # ── Stage 7: Keypoints ────────────────────────────────────────
        kps_3d = np.zeros((0, 3), dtype=np.float32)

        if mask is not None and depth_map is not None:
            if self.cfg['keypoints'].get('use_geometric', True):
                kps_3d, kps_2d = self.geo_keypoints.extract_3d(
                    frame, mask, depth_map, self.depth_localizer
                )
                result['kps_2d'] = kps_2d

            sem_kps_2d = self.sem_keypoints.detect_rule_based(
                det['class_name'], mask, frame
            )
            sem_kps_3d = self.sem_keypoints.lift_keypoints_to_3d(
                sem_kps_2d, depth_map, self.depth_localizer
            )
            result['sem_kps_3d'] = sem_kps_3d

            kp_name, kp_pos = self.sem_keypoints.get_grasp_keypoint(
                det['class_name'], sem_kps_3d
            )
            result['grasp_keypoint_name'] = kp_name
            result['grasp_keypoint_3d']   = kp_pos

        result['kps_3d'] = kps_3d

        # ── Build object point cloud for AnyGrasp ─────────────────────
        pointcloud = np.zeros((0, 3), dtype=np.float32)

        if mask is not None and depth_map is not None:
            kernel       = np.ones((10, 10), np.uint8)
            # The outer guard guarantees mask is not None here —
            # no ternary needed.
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            # build_object_pointcloud returns a single Nx3 array.
            pointcloud   = self.geo_keypoints.build_object_pointcloud(
                dilated_mask, depth_map, self.depth_localizer,
                max_points=1024,
            )

        result['pointcloud'] = pointcloud

        t4 = time.perf_counter()
        print(f"  [timing] keypoints + pointcloud: {(t4 - t3)*1000:.1f} ms")
        print(f"  [timing] total perception: {(t4 - t_stage)*1000:.1f} ms")

        return result

    # ─────────────────────────────────────────────────────────────────
    # Grasp planning
    # ─────────────────────────────────────────────────────────────────

    def plan_grasp(self, perception_result):
        """
        Given a perception result, plan grasp poses using AnyGrasp.

        Returns list of grasp candidates sorted by score descending,
        or [] on failure.

        If AnyGrasp returns nothing and an object angle is available
        from mask PCA, retries with the TopDown fallback planner using
        the angle for gripper orientation.
        """
        pcd    = perception_result['pointcloud']
        center = perception_result['object_3d']
        angle  = perception_result.get('angle')

        # Prefer semantic keypoint as grasp target if available.
        kp_pos = perception_result.get('grasp_keypoint_3d')
        if kp_pos is not None:
            center = kp_pos

        grasps = self.grasp_planner.plan(
            points_xyz       = pcd,
            object_center_3d = center,
            top_k            = self.cfg['anygrasp'].get('top_k', 5),
        )

        # If AnyGrasp fell back to top-down internally, angle was not
        # passed (AnyGraspPlanner.plan() has no angle param). Re-call
        # fallback explicitly with angle so the gripper is aligned to
        # the object's principal axis rather than defaulting to 0°.
        if not grasps and angle is not None:
            grasps = self.grasp_planner.fallback.plan(
                center,
                top_k        = self.cfg['anygrasp'].get('top_k', 5),
                object_angle = angle,
            )

        if not grasps:
            print("GRASP PLANNING FAIL: no valid grasps found")

        return grasps

    # ─────────────────────────────────────────────────────────────────
    # Single grasp attempt
    # ─────────────────────────────────────────────────────────────────

    def run_single_grasp(self, target_class=None, offset=None, visualize=True):
        """
        Execute one complete grasp attempt.

        Returns True on success.
        """
        print(f"\nGrasp attempt — target: {target_class or 'any'}")

        # Flush camera buffer. Track the last valid frame explicitly —
        # if all reads fail, frame stays None rather than holding a
        # stale value from a previous loop iteration.
        frame = None
        for _ in range(5):
            f = self.get_frame()
            if f is not None:
                frame = f
        if frame is None:
            print("FAIL: no camera frame")
            return False

        # ── Perception ────────────────────────────────────────────────
        perc = self.run_perception(frame, target_class)
        if perc is None:
            return False

        print(f"  Object: {perc['detection']['class_name']} "
              f"(conf={perc['detection']['confidence']:.2f})")
        print(f"  3D position: {perc['object_3d'] * 100} cm")
        print(f"  Point cloud: {len(perc['pointcloud'])} points")
        print(f"  Keypoints:   {len(perc['kps_3d'])} geometric, "
              f"{len(perc.get('sem_kps_3d', {}))} semantic")

        # ── Apply retry offset ─────────────────────────────────────────
        if offset is not None:
            # Validate offset before applying — a buggy retry strategy
            # could produce NaN/Inf which would silently corrupt 3D positions.
            offset = np.asarray(offset, dtype=np.float64)
            if not np.all(np.isfinite(offset)):
                print(f"WARNING: retry offset contains non-finite values "
                      f"{offset} — ignoring offset.")
                offset = None

        if offset is not None:
            perc['object_3d'] = perc['object_3d'] + offset
            if len(perc['pointcloud']) > 0:
                perc['pointcloud'] = (
                    perc['pointcloud'] + offset.astype(np.float32)
                )

        # ── Grasp planning ────────────────────────────────────────────
        grasps = self.plan_grasp(perc)
        if not grasps:
            return False

        best_grasp = grasps[0]
        print(f"  Best grasp: score={best_grasp['score']:.3f} "
              f"source={best_grasp['source']}")

        # position_base / rotation_base may be None if T_cam_to_base was
        # not calibrated (anygrasp_planner.py sets them to None in that case).
        if (best_grasp['position_base'] is None
                or best_grasp['rotation_base'] is None):
            print("FAIL: grasp base-frame pose is None — "
                  "run hand_eye_calibration.py first.")
            return False

        # ── Pre-grasp motion ──────────────────────────────────────────
        pre_grasp_pos = best_grasp.get('pre_grasp_base')

        # pre_grasp_base should always be populated by anygrasp_planner.py.
        # This fallback only fires if the key is missing entirely — warn so
        # you notice. The naive +Z offset is wrong for side grasps.
        if pre_grasp_pos is None:
            print("WARNING: pre_grasp_base missing from grasp dict — "
                  "using naive +Z fallback. Verify grasp planner output.")
            pre_grasp_pos = (best_grasp['position_base']
                             + np.array([0.0, 0.0, 0.15]))

        R_base   = best_grasp['rotation_base']
        orn_quat = Rotation.from_matrix(R_base).as_quat()

        print(f"  Moving to pre-grasp: {pre_grasp_pos * 100} cm")
        success = self.motion_planner.move_to_pose(
            pre_grasp_pos, orn_quat,
            velocity_scale=self.cfg['motion_planning']['velocity_scale'],
        )
        if not success:
            print("FAIL: motion planning to pre-grasp failed")
            return False

        # Wait for arm to fully settle before capturing the reference frame.
        # Capturing during motion gives a blurred frame that hurts verify_by_vision.
        time.sleep(1.2)
        frame_before = self.get_frame()

        # ── Visual servo ──────────────────────────────────────────────
        def frame_gen():
            while True:
                f = self.get_frame()
                if f is not None:
                    yield f

        servo_ok = self.servo.run_grasp_sequence(
            frame_gen(),
            self.detector,
            self.gripper_tracker,
            self._make_localizer(
                perc['depth_map'],
                perc['tag']['tvec'],
                perc['tag']['center_2d'],
            ),
            target_class=target_class,
            visualize=visualize,
        )

        if not servo_ok:
            print("FAIL: visual servo did not converge")
            return False

        # ── Verify and lift ───────────────────────────────────────────
        time.sleep(0.3)
        frame_after = self.get_frame()

        current_ok = self.verifier.verify_by_current()

        # verify_by_vision crashes on None frames — guard explicitly.
        visual_ok = (
            self.verifier.verify_by_vision(
                frame_before, frame_after,
                self.detector, target_class,
            )
            if frame_before is not None and frame_after is not None
            else False
        )

        if current_ok or visual_ok:
            print(f"  SUCCESS (current={current_ok}, visual={visual_ok})")
            return True
        else:
            print(f"  FAIL: grasp not verified "
                  f"(current={current_ok}, visual={visual_ok})")
            return False

    def _make_localizer(self, depth_map, tag_tvec, tag_center_2d):
        """
        Returns a localizer stub the visual servo can use to convert
        pixel coordinates to 3D positions during the servo loop.

        Uses the depth map captured at grasp-plan time as an approximation.
        For higher accuracy, recompute depth_map each servo frame.
        """
        localizer = self.depth_localizer

        class _Localizer:
            def pixel_to_3d(self, u, v):
                if depth_map is None:
                    return None
                return localizer.pixel_to_3d(u, v, depth_map)

            def is_calibrated(self):
                return depth_map is not None

        return _Localizer()

    # ─────────────────────────────────────────────────────────────────
    # Retry loop
    # ─────────────────────────────────────────────────────────────────

    def run_with_retry(self, target_class=None, visualize=True):
        """Execute grasp with automatic retry on failure."""
        def grasp_fn(offset):
            return self.run_single_grasp(
                target_class=target_class,
                offset=offset,
                visualize=visualize,
            )

        def verify_fn():
            return self.verifier.verify_by_current()

        return self.retry.run(grasp_fn, verify_fn, self.arm)

    # ─────────────────────────────────────────────────────────────────
    # Operating modes
    # ─────────────────────────────────────────────────────────────────

    def run_continuous(self, target_class=None):
        """
        Repeatedly attempt grasps until KeyboardInterrupt.
        Reports running success rate.
        """
        attempt              = 0
        successes            = 0
        consecutive_failures = 0
        max_consecutive      = self.cfg.get('pipeline', {}).get(
            'max_consecutive_failures', 5
        )

        print("Continuous mode. Ctrl+C to stop.\n")
        try:
            while True:
                attempt += 1
                print(f"\n{'='*40}  ATTEMPT {attempt}  {'='*40}")
                success = self.run_with_retry(target_class)

                if success:
                    successes            += 1
                    consecutive_failures  = 0
                else:
                    consecutive_failures += 1

                rate = 100 * successes / attempt
                print(f"Running rate: {successes}/{attempt} ({rate:.0f}%)")

                if consecutive_failures >= max_consecutive:
                    print(f"\nWARNING: {consecutive_failures} consecutive "
                          f"failures. Moving home and pausing for 5s.")
                    print("Check: lighting, tag visibility, object presence.")
                    try:
                        self.motion_planner.move_home()
                    except Exception as e:
                        print(f"move_home failed during recovery: {e}")
                    time.sleep(5.0)
                    consecutive_failures = 0
                else:
                    # Correct order: move home FIRST, then open gripper.
                    # Opening gripper before moving home drops the object
                    # immediately — possibly onto the robot or user.
                    try:
                        self.motion_planner.move_home()
                    except Exception as e:
                        print(f"move_home failed: {e}")
                    time.sleep(0.3)
                    self.arm.open_gripper()
                    time.sleep(2.0)

        except KeyboardInterrupt:
            print(f"\nFinal: {successes}/{attempt} successful")
        finally:
            self.shutdown()

    def run_dryrun(self, target_class=None):
        """
        Dry-run mode — runs perception + grasp planning only.
        No arm movement. Use this to tune perception on the bench.
        """
        print("=== DRY RUN MODE — no arm movement ===\n")

        frame = None
        for _ in range(5):
            f = self.get_frame()
            if f is not None:
                frame = f
        if frame is None:
            print("FAIL: no camera frame")
            return

        perc = self.run_perception(frame, target_class)
        if perc is None:
            print("Perception failed — check tag visibility and lighting")
            return

        print("\nPerception result:")
        print(f"  Class:        {perc['detection']['class_name']} "
              f"(conf={perc['detection']['confidence']:.2f})")
        print(f"  Camera frame: {perc['object_3d'] * 100} cm")
        print(f"  Point cloud:  {len(perc['pointcloud'])} points")
        print(f"  Geo keypts:   {len(perc['kps_3d'])}")
        print(f"  Sem keypts:   {list(perc.get('sem_kps_3d', {}).keys())}")

        grasps = self.plan_grasp(perc)
        if not grasps:
            print("\nGrasp planning failed")
            return

        print(f"\nGrasp candidates: {len(grasps)}")
        for i, g in enumerate(grasps):
            # position_base may be None when T_cam_to_base is not calibrated.
            pos_str = (
                f"{g['position_base'] * 100} cm"
                if g['position_base'] is not None
                else "N/A (T_cam_to_base not calibrated)"
            )
            print(f"  [{i}] score={g['score']:.3f}  "
                  f"pos_base={pos_str}  source={g['source']}")

        print("\nDry run complete — no arm movement executed.")

    def calibration_mode(self):
        """Run all calibration procedures in sequence."""
        print("=== CALIBRATION MODE ===\n")

        print("Step 1: Camera intrinsic calibration")
        from calibration.run_calibration import run_calibration
        run_calibration()

        print("\nStep 2: Hand-eye calibration (camera-to-base transform)")
        from calibration.hand_eye_calibration import HandEyeCalibrator
        calibrator = HandEyeCalibrator(self.config_path)
        R_g2b, t_g2b, R_t2c, t_t2c = calibrator.collect_poses(
            self.arm, self.cfg['camera']['device_index']
        )
        calibrator.solve(R_g2b, t_g2b, R_t2c, t_t2c)

        if self.cfg['localization'].get('use_table_plane', False):
            print("\nStep 3: Table plane calibration (optional)")
            from perception.localization.table_plane import TablePlaneLocalizer
            tpl = TablePlaneLocalizer(self.config_path)
            tpl.calibrate_plane(
                self.gripper_tracker,
                self.cfg['camera']['device_index'],
            )

        print("\nAll calibrations complete.")

    def diagnostic_mode(self):
        """
        Live diagnostic view showing all pipeline stages.
        Shows: raw frame | detections | depth map (side by side)
        Press ESC to exit.
        """
        print("Diagnostic mode — ESC to exit")
        print("Shows: raw frame | detections | depth map\n")

        # Load T_cam_to_base once before the loop — not per frame.
        # np.load does a file syscall; doing it at 30fps is wasteful.
        t_cam_path    = self.cfg['transforms']['T_cam_to_base_file']
        T_cam_to_base = None
        if Path(t_cam_path).exists():
            T_cam_to_base = np.load(t_cam_path)
            print("T_cam_to_base: loaded")
        else:
            print("T_cam_to_base: NOT calibrated")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            gripper_det = self.gripper_tracker.detect(frame)
            if gripper_det is not None:
                self.last_tag_detection = gripper_det
                self.last_tag_time      = time.time()

            detections = self.detector.detect_objects(frame)

            display = frame.copy()
            display = self.gripper_tracker.draw_overlay(display, gripper_det)
            if detections:
                display = self.detector.draw_detections(display, detections)

            depth_map = None
            if (self.depth_localizer.depth_model is not None
                    and gripper_det is not None):
                depth_map = self.depth_localizer.get_scaled_depth_map(
                    frame,
                    gripper_det['tvec'],
                    gripper_det['center_2d'],
                )
                if depth_map is not None:
                    d_vis   = np.clip(depth_map, 0, 2.0)
                    d_vis   = (d_vis / 2.0 * 255).astype(np.uint8)
                    d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_PLASMA)

                    # np.hstack requires matching height. If the depth map
                    # was produced at a different resolution, resize to match.
                    if d_color.shape[0] != display.shape[0]:
                        d_color = cv2.resize(
                            d_color,
                            (int(d_color.shape[1] * display.shape[0]
                                 / d_color.shape[0]),
                             display.shape[0]),
                        )
                    display = np.hstack([display, d_color])

            y       = 30
            tag_src = gripper_det or self.last_tag_detection
            if tag_src:
                p_cam    = tag_src['tvec'] * 100
                info_cam = (f"Gripper cam: ({p_cam[0]:.1f}, {p_cam[1]:.1f},"
                            f" {p_cam[2]:.1f}) cm | "
                            f"depth: {'OK' if self.depth_localizer.depth_model else 'MISSING'}")
                cv2.putText(display, info_cam, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 22

                if T_cam_to_base is not None:
                    tvec_h = np.append(tag_src['tvec'], 1.0)
                    p_base = (T_cam_to_base @ tvec_h)[:3] * 100
                    cv2.putText(
                        display,
                        f"Gripper base: ({p_base[0]:.1f}, "
                        f"{p_base[1]:.1f}, {p_base[2]:.1f}) cm",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
                    )
                    y += 22

                    if detections and depth_map is not None:
                        detections.sort(key=lambda d: d['confidence'],
                                        reverse=True)
                        cx_px, cy_px = detections[0]['center_2d']
                        obj_cam = self.depth_localizer.pixel_to_3d(
                            cx_px, cy_px, depth_map
                        )
                        if obj_cam is not None:
                            obj_h    = np.append(obj_cam, 1.0)
                            obj_base = (T_cam_to_base @ obj_h)[:3] * 100
                            cv2.putText(
                                display,
                                (f"Object cam: ({obj_cam[0]*100:.1f},"
                                 f" {obj_cam[1]*100:.1f},"
                                 f" {obj_cam[2]*100:.1f}) cm"),
                                (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1,
                            )
                            y += 22
                            cv2.putText(
                                display,
                                (f"Object base: ({obj_base[0]:.1f},"
                                 f" {obj_base[1]:.1f},"
                                 f" {obj_base[2]:.1f}) cm"),
                                (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                            )
                            y += 22
                else:
                    cv2.putText(display, "T_cam_to_base: NOT calibrated",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                    y += 22

            for det in detections[:3]:
                cv2.putText(display,
                            f"{det['class_name']} {det['confidence']:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 165, 0), 1)
                y += 22

            scale = min(1.0, 1600 / display.shape[1])
            if scale < 1.0:
                display = cv2.resize(display, None, fx=scale, fy=scale)

            cv2.imshow('Diagnostic', display)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    # ─────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────

    def shutdown(self):
        """
        Clean shutdown — each subsystem wrapped independently so a failure
        in one doesn't skip cleanup of the others.
        """
        print("Shutting down...")
        for name, fn in [
            ("arm",                   lambda: self.arm.shutdown()),
            ("motion_planner",        lambda: self.motion_planner.shutdown()),
            ("cap.release",           lambda: self.cap.release()),
            ("cv2.destroyAllWindows", cv2.destroyAllWindows),
        ]:
            try:
                fn()
            except Exception as e:
                print(f"{name} shutdown error (ignored): {e}")
        print("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Wheelchair Arm — Grasp Pipeline'
    )
    parser.add_argument(
        '--mode', default='grasp',
        choices=['grasp', 'calibrate', 'diagnostic', 'continuous', 'dryrun'],
        help='Operating mode',
    )
    parser.add_argument(
        '--class', dest='target_class', default=None,
        help='Target class name (e.g. bottle, cup). None = any.',
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Disable visualization windows',
    )
    parser.add_argument(
        '--config', default='configs/config.yaml',
        help='Path to config.yaml',
    )
    args = parser.parse_args()

    pipeline = WheelchairArmPipeline(args.config)

    try:
        if args.mode == 'calibrate':
            pipeline.calibration_mode()

        elif args.mode == 'diagnostic':
            pipeline.diagnostic_mode()

        elif args.mode == 'continuous':
            pipeline.run_continuous(args.target_class)

        elif args.mode == 'dryrun':
            pipeline.run_dryrun(args.target_class)

        else:   # grasp
            ok = pipeline.run_with_retry(
                target_class=args.target_class,
                visualize=not args.no_viz,
            )
            print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")

    finally:
        pipeline.shutdown()