# planning/servo/visual_servo.py
import cv2
import numpy as np
import time
import yaml


class VisualServoController:
    """
    Image-Based Visual Servoing (IBVS):
    Controls arm motion to minimize error between
    object position and gripper position in the image.

    Uses AprilTag as gripper reference.
    No IK or motion planning required for approach phase.

    Control law:
        v = Kp * e
        where e = object_metric - gripper_metric (lateral, converted from pixels)
              e = object_depth - gripper_depth (axial)

    Coordinate frame:
        Velocity output is in OpenCV camera frame (+X right, +Y down, +Z forward).
        Caller must transform through T_cam_to_base before sending to arm.
        See hand_eye_calibration.py for that transform.
    """

    def __init__(self, arm_controller, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Load camera intrinsics for pixel → metric error conversion
        with open(cfg['camera']['calibration_file']) as f:
            calib_data = yaml.safe_load(f)
        self.K  = np.array(calib_data['camera_matrix']['data'],
                           dtype=np.float64).reshape(3, 3)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]

        sv = cfg['servo']
        self.Kp_xy    = sv['Kp_xy']             # control gain for lateral (metric)
        self.Kp_z     = sv['Kp_z']              # control gain for depth
        self.xy_thresh = sv['xy_threshold_pixels']  # convergence threshold (meters)
        self.z_thresh  = sv['z_threshold_meters']
        self.max_vel   = sv['max_velocity_ms']
        self.timeout   = sv['timeout_seconds']

        self.arm = arm_controller

        # EMA filter state — applied BEFORE velocity computation
        self._filtered_err = np.zeros(3)    # [ex_m, ey_m, ez_m]
        self._ema_alpha    = 0.2            # weight for new measurement

        # State machine
        self.STATES = ['IDLE', 'APPROACH_XY', 'APPROACH_Z',
                       'GRASP', 'LIFT', 'DONE', 'FAILED']
        self.state = 'IDLE'

        # Miss counters — abort if detection lost for too long
        self._object_miss_count  = 0
        self._gripper_miss_count = 0
        self.max_miss_frames     = 10

        # Last known errors — stored for visualization block
        # (visualization runs before state machine so errors must be pre-stored)
        self._last_err_xy_m = 0.0
        self._last_err_z_m  = 0.0

    # ── Control law ──────────────────────────────────────────────────────────────

    def compute_velocity(self, object_2d, gripper_2d,
                         object_depth, gripper_depth):
        """
        Compute velocity command to move gripper toward object.

        Lateral (XY): pixel error converted to metric, EMA filtered, then Kp applied.
        Axial  (Z):   depth error in meters, EMA filtered, then Kp applied.

        EMA is applied BEFORE velocity computation so the arm receives
        smoothed commands, not just smoothed error readouts.

        Returns:
            vel:   [vx, vy, vz] in camera frame (m/s), clipped to max_vel
            err_x: filtered lateral error X (meters)
            err_y: filtered lateral error Y (meters)
            err_z: filtered axial error   Z (meters)
        """
        # Raw pixel → metric error conversion using gripper depth as Z reference
        Z     = float(gripper_depth)
        raw_x = float(object_2d[0] - gripper_2d[0]) * Z / self.fx   # meters
        raw_y = float(object_2d[1] - gripper_2d[1]) * Z / self.fy   # meters
        raw_z = float(object_depth - gripper_depth)                   # meters

        # Apply EMA FIRST — velocity is computed from filtered errors
        raw_err            = np.array([raw_x, raw_y, raw_z])
        self._filtered_err = (self._ema_alpha * raw_err +
                              (1.0 - self._ema_alpha) * self._filtered_err)
        err_x, err_y, err_z = self._filtered_err

        # Proportional control on filtered errors
        vx  = self.Kp_xy * err_x
        vy  = self.Kp_xy * err_y
        vz  = self.Kp_z  * err_z
        vel = np.array([vx, vy, vz])

        # Clip to max velocity (preserves direction)
        norm = np.linalg.norm(vel)
        if norm > self.max_vel:
            vel = vel * self.max_vel / norm

        return vel, err_x, err_y, err_z

    # ── Convergence checks ───────────────────────────────────────────────────────

    def is_xy_converged(self, err_x, err_y):
        return np.sqrt(err_x**2 + err_y**2) < self.xy_thresh

    def is_z_converged(self, err_z):
        return abs(err_z) < self.z_thresh

    # ── Main grasp loop ──────────────────────────────────────────────────────────

    def run_grasp_sequence(self, frame_source, object_detector,
                           gripper_tracker, localizer,
                           target_class=None, visualize=True):
        """
        Full grasp state machine.

        Args:
            frame_source:    callable or generator returning BGR frames
            object_detector: ObjectDetector instance
            gripper_tracker: GripperTracker instance
            localizer:       ScaledDepthLocalizer instance
            target_class:    string class name to target, or None for any
            visualize:       show live visualization window

        Returns:
            True if grasp succeeded, False otherwise
        """
        self.state  = 'APPROACH_XY'
        start_time  = time.time()

        consecutive_converged = 0
        required_convergence  = 15   # ~0.5s at 30fps — robust to noisy detections

        # Reset miss counters at start of each grasp attempt
        self._object_miss_count  = 0
        self._gripper_miss_count = 0

        print("Starting visual servo grasp sequence...")

        while time.time() - start_time < self.timeout:

            # ── Get frame ────────────────────────────────────────────────────────
            frame = frame_source() if callable(frame_source) else next(frame_source)
            if frame is None:
                continue

            # ── Detect gripper tag ───────────────────────────────────────────────
            gripper_det = gripper_tracker.detect(frame)
            if gripper_det is None:
                self._gripper_miss_count += 1
                if self._gripper_miss_count > self.max_miss_frames:
                    print("Gripper tag lost — aborting")
                    self.state = 'FAILED'
                    self.arm.stop()
                    break
                continue
            self._gripper_miss_count = 0   # reset on successful detection

            # ── Detect object ────────────────────────────────────────────────────
            object_dets = object_detector.detect_objects(frame)
            if not object_dets:
                self._object_miss_count += 1
                if self._object_miss_count > self.max_miss_frames:
                    print("Object lost — aborting")
                    self.state = 'FAILED'
                    self.arm.stop()
                    break
                continue
            self._object_miss_count = 0    # reset on successful detection

            # Filter by class if specified
            if target_class:
                object_dets = [d for d in object_dets
                               if d['class_name'] == target_class]
                if not object_dets:
                    continue

            object_det = object_dets[0]

            # ── 2D centres ───────────────────────────────────────────────────────
            object_2d  = object_det['center_2d']
            gripper_2d = gripper_det['center_2d']

            # ── Depths ───────────────────────────────────────────────────────────
            gripper_depth = gripper_det['tvec'][2]   # PnP Z in camera frame (m)

            object_pos_3d = localizer.pixel_to_3d(
                object_2d[0], object_2d[1]
            )
            if object_pos_3d is None:
                print("Warning: could not localize object in 3D")
                continue

            object_depth = object_pos_3d[2]

            # ── Visualization ────────────────────────────────────────────────────
            if visualize:
                display = frame.copy()
                display = gripper_tracker.draw_overlay(display, gripper_det)
                display = object_detector.draw_detections(display, [object_det])

                g2d = gripper_2d.astype(int)
                o2d = object_2d.astype(int)
                cv2.line(display, tuple(g2d), tuple(o2d), (255, 165, 0), 2)

                # Errors stored from previous state machine iteration
                # displayed in cm for readability
                cv2.putText(display,
                            f"State: {self.state}  "
                            f"err_xy={self._last_err_xy_m * 100:.1f}cm  "
                            f"err_z={self._last_err_z_m * 100:.1f}cm",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

                cv2.imshow('Visual Servo', display)
                if cv2.waitKey(1) == 27:    # ESC
                    self.arm.stop()
                    if visualize:
                        cv2.destroyAllWindows()
                    return False

            # ── State machine ────────────────────────────────────────────────────

            if self.state == 'APPROACH_XY':
                vel, ex, ey, ez = self.compute_velocity(
                    object_2d, gripper_2d, object_depth, gripper_depth
                )
                self._last_err_xy_m = np.sqrt(ex**2 + ey**2)
                self._last_err_z_m  = ez

                vel[2] = 0.0    # zero axial velocity — lateral centering only
                self.arm.send_velocity(vel)

                if self.is_xy_converged(ex, ey):
                    consecutive_converged += 1
                    if consecutive_converged >= required_convergence:
                        print("XY converged — approaching in depth")
                        self.state = 'APPROACH_Z'
                        consecutive_converged = 0
                else:
                    consecutive_converged = 0

            elif self.state == 'APPROACH_Z':
                vel, ex, ey, ez = self.compute_velocity(
                    object_2d, gripper_2d, object_depth, gripper_depth
                )
                self._last_err_xy_m = np.sqrt(ex**2 + ey**2)
                self._last_err_z_m  = ez

                self.arm.send_velocity(vel)

                if self.is_xy_converged(ex, ey) and self.is_z_converged(ez):
                    consecutive_converged += 1
                    if consecutive_converged >= required_convergence:
                        self.arm.stop()
                        print("Converged at grasp position")
                        self.state = 'GRASP'
                        consecutive_converged = 0
                elif not self.is_xy_converged(ex, ey):
                    # Drifted laterally — return to centering phase
                    self.state = 'APPROACH_XY'
                    consecutive_converged = 0

            elif self.state == 'GRASP':
                time.sleep(0.2)
                self.arm.close_gripper()
                print("Gripper closing...")
                time.sleep(1.0)

                if self.arm.grasp_detected():
                    print("Contact detected — grasp successful")
                    self.state = 'LIFT'
                else:
                    print("No contact detected — grasp failed")
                    self.state = 'FAILED'
                    self.arm.stop()

            elif self.state == 'LIFT':
                print("Lifting...")
                self.arm.lift(height_m=0.12)
                self.state = 'DONE'
                if visualize:
                    cv2.destroyAllWindows()
                return True

        # ── Timeout ──────────────────────────────────────────────────────────────
        self.arm.stop()
        print(f"Visual servo timed out after {self.timeout}s")
        self.state = 'FAILED'
        if visualize:
            cv2.destroyAllWindows()
        return False