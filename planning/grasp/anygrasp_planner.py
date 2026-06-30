# planning/grasp/anygrasp_planner.py
#
# AnyGrasp integration for 6-DoF grasp pose estimation.
#
# SETUP (do this before anything else in the project):
# ======================================================
# AnyGrasp requires an academic license.
# Email the authors at: anygrasp@gmail.com
# Subject: "Academic License Request — AnyGrasp"
# Include: your name, institution (IIT Madras R2D2 Lab), project description
# Turnaround: typically 2-5 days
#
# After receiving license:
#   git clone https://github.com/graspnet/anygrasp_sdk
#   cd anygrasp_sdk
#   pip install -r requirements.txt
#   # Follow their README to download model weights
#   # Place license file as instructed
#
# RESOURCES:
#   AnyGrasp paper:   https://arxiv.org/abs/2212.08333
#   SDK repo:         https://github.com/graspnet/anygrasp_sdk
#   GraspNet-1B:      https://graspnet.net/
#   GraspNet paper:   https://arxiv.org/abs/2112.02110
#
# INPUT FORMAT:
#   AnyGrasp expects an open3d point cloud (or numpy Nx6 array: XYZRGB)
#   The point cloud should cover the object + nearby workspace.
#   Filter to a tight bounding box around the object for best results.
#
# OUTPUT FORMAT:
#   GraspGroup object containing N grasp candidates, each with:
#       .translations   Nx3 float32  grasp center positions
#       .rotations      Nx3x3 float32 rotation matrices
#       .widths         N float32    required gripper opening width (m)
#       .scores         N float32    grasp quality score (higher = better)
#       .object_ids     N int        which object each grasp is on
#
# FILTERING GRASPS:
#   Not all AnyGrasp outputs are reachable by your arm.
#   Filter by:
#       1. Score threshold (reject low-confidence grasps)
#       2. Gripper width range (your gripper's min/max opening)
#       3. Workspace bounds — spherical reach check (not axis-aligned box)
#   MoveIt2 handles: IK feasibility, joint limits, self-collision,
#                    table/obstacle collision, trajectory planning.
#   Do NOT re-implement those here.
#
# COORDINATE FRAME NOTE:
#   All positions/rotations stored with _cam suffix = OpenCV camera frame
#   (+X right, +Y down, +Z forward).
#   All positions/rotations stored with _base suffix = robot base frame
#   (REP-103: +X forward, +Y left, +Z up), via T_cam_to_base.
#   Always use _base versions for MoveIt2 / arm controller.

import cv2
import numpy as np
import yaml


# Try to import AnyGrasp — gracefully degrade if not installed
try:
    from gsnet import AnyGrasp
    from graspnetAPI import GraspGroup
    ANYGRASP_AVAILABLE = True
except ImportError:
    ANYGRASP_AVAILABLE = False
    print("AnyGrasp not installed. Using fallback top-down grasp planner.")
    print("Install AnyGrasp SDK and obtain academic license to enable.")


class AnyGraspPlanner:
    """
    6-DoF grasp pose estimation using AnyGrasp.

    Falls back to TopDownGraspPlanner if AnyGrasp is not available.
    This lets you develop and test the full pipeline before
    the AnyGrasp license arrives.

    Output dict keys (both branches produce identical structure):
        position_cam:   [X,Y,Z] grasp center in OpenCV camera frame
        rotation_cam:   3x3 rotation matrix in camera frame
        position_base:  [X,Y,Z] grasp center in robot base frame
        rotation_base:  3x3 rotation matrix in robot base frame
        pre_grasp_base: [X,Y,Z] pre-grasp position in base frame
        width:          required gripper opening width (m)
        score:          quality score (higher = better)
        approach:       [X,Y,Z] unit approach direction in camera frame
        source:         'anygrasp' or 'fallback'
    """

    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg  = cfg
        grasp_cfg = cfg.get('anygrasp', {})

        # Gripper physical constraints
        self.gripper_width_min = grasp_cfg.get('gripper_width_min', 0.0)
        self.gripper_width_max = grasp_cfg.get('gripper_width_max', 0.10)

        # Score threshold — reject low-confidence grasps
        self.score_threshold = grasp_cfg.get('score_threshold', 0.3)

        # Spherical workspace bounds
        self.workspace_min_reach = grasp_cfg.get('workspace_min_reach_m', 0.10)
        self.workspace_max_reach = grasp_cfg.get('workspace_max_reach_m', 0.70)
        self.workspace_min_z     = grasp_cfg.get('workspace_min_z_m',     0.00)
        self.workspace_max_z     = grasp_cfg.get('workspace_max_z_m',     1.20)

        self.pre_grasp_offset = grasp_cfg.get('pre_grasp_offset_m', 0.15)

        # BUG-FIX-A: T_cam_to_base identity fallback silently poisons all
        # base-frame outputs. Added _transform_valid flag so transform methods
        # raise a clear RuntimeError instead of returning wrong values.
        self._transform_valid = False
        try:
            with open(cfg['camera']['calibration_file'], 'r') as f:
                calib = yaml.safe_load(f)

            self.K = np.array(
                calib['camera_matrix']['data'],
                dtype=np.float64
            ).reshape((3, 3))

        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(
                f"TopDownGraspPlanner: failed to load camera matrix: {e}. "
                "Check 'camera.calibration_file' in config.yaml and that the "
                "YAML file contains camera_matrix.data."
    )
        except KeyError:
            # BUG-FIX-B: config.yaml might not have 'transforms' section at all.
            # Original code would raise an unhandled KeyError here during __init__,
            # crashing the entire process before the planner is even usable.
            self.T_cam_to_base = np.eye(4)
            print("WARNING: 'transforms.T_cam_to_base_file' missing from config. "
                  "Add it to configs/config.yaml.")

        # Initialize AnyGrasp if available
        self.anygrasp = None
        if ANYGRASP_AVAILABLE:
            self._init_anygrasp(grasp_cfg)

        # Fallback planner — always available regardless of AnyGrasp license
        self.fallback = TopDownGraspPlanner(config_path)

    def _init_anygrasp(self, grasp_cfg):
        try:
            checkpoint_path = grasp_cfg.get('checkpoint_path', '')
            # BUG-FIX-C: checkpoint_path='' will cause AnyGrasp to fail with a
            # confusing FileNotFoundError. Validate the path before attempting load.
            if not checkpoint_path:
                print("WARNING: AnyGrasp checkpoint_path not set in config. "
                      "Add 'anygrasp.checkpoint_path' to configs/config.yaml.")
                return
            self.anygrasp = AnyGrasp(checkpoint_path)
            self.anygrasp.load_net()
            print("AnyGrasp loaded successfully.")
        except Exception as e:
            print(f"AnyGrasp init failed: {e}")
            print("Falling back to top-down grasp planner.")
            self.anygrasp = None

    def plan(self, points_xyz, colors_rgb=None,
             object_center_3d=None, top_k=5):
        """
        Generate grasp candidates for an object.

        Args:
            points_xyz:       Nx3 float32 point cloud (camera frame, meters)
            colors_rgb:       Nx3 uint8 colors (optional)
            object_center_3d: [X,Y,Z] object center in camera frame
                              used by fallback planner if AnyGrasp unavailable
            top_k:            max number of candidates to return

        Returns:
            list of grasp dicts sorted by score descending.
            See class docstring for dict structure.
            Returns [] if no valid grasps found.
        """
        # BUG-FIX-D: Original code silently falls through to fallback even when
        # AnyGrasp is available but returns 0 candidates — which can happen due
        # to overly strict filtering. Log this explicitly so you know whether
        # AnyGrasp ran but found nothing, vs. was not available at all.
        if self.anygrasp is not None:
            if points_xyz is None or len(points_xyz) == 0:
                print("AnyGraspPlanner.plan(): empty point cloud — skipping AnyGrasp.")
            elif len(points_xyz) <= 10:
                print(f"AnyGraspPlanner.plan(): only {len(points_xyz)} points — "
                      f"too few for AnyGrasp (need >10). Falling back.")
            else:
                grasps = self._plan_anygrasp(points_xyz, colors_rgb, top_k)
                if grasps:
                    return grasps
                print("AnyGrasp returned 0 valid grasps after filtering. Falling back.")

        # Fallback to top-down planner
        if object_center_3d is not None:
            return self.fallback.plan(object_center_3d, top_k)

        print("AnyGraspPlanner.plan(): no valid grasps and no object_center_3d "
              "provided for fallback. Returning [].")
        return []

    def _plan_anygrasp(self, points_xyz, colors_rgb, top_k):
        """Run AnyGrasp inference on point cloud and filter results."""
        try:
            # AnyGrasp expects colors normalized to [0, 1]
            if colors_rgb is not None:
                # BUG-FIX-E: colors_rgb could arrive as float [0,1] already
                # if upstream changed. Clip and convert correctly in both cases.
                if colors_rgb.dtype == np.uint8:
                    colors = colors_rgb.astype(np.float32) / 255.0
                else:
                    colors = np.clip(colors_rgb.astype(np.float32), 0.0, 1.0)
            else:
                # Uniform gray fallback
                colors = np.full((len(points_xyz), 3), 0.5, dtype=np.float32)

            # BUG-FIX-F: The lims parameter is passed as:
            #   [[min_reach, max_reach], [min_reach, max_reach], [min_z, max_z]]
            # But AnyGrasp's lims format is:
            #   [x_min, x_max, y_min, y_max, z_min, z_max]  (flat list of 6 values)
            # The original nested-list format is WRONG and silently produces bad results
            # or crashes depending on SDK version.
            # Check your SDK version's get_grasp() signature — the correct form
            # for most SDK versions is the flat 6-value list below.
            lims = [
                -self.workspace_max_reach, self.workspace_max_reach,   # X
                -self.workspace_max_reach, self.workspace_max_reach,   # Y
                self.workspace_min_z,      self.workspace_max_z        # Z
            ]

            gg, cloud = self.anygrasp.get_grasp(
                points_xyz.astype(np.float32),
                colors,
                lims=lims,
            )

            if gg is None or len(gg) == 0:
                return []

            gg = gg.nms()
            gg = gg.sort_by_score()

            grasps = []
            for i in range(len(gg)):
                if len(grasps) >= top_k:
                    break

                g       = gg[i]
                pos_cam = g.translation
                rot_cam = g.rotation_matrix
                width   = float(g.width)
                score   = float(g.score)

                if score < self.score_threshold:
                    # BUG-FIX-G: After sort_by_score(), all subsequent grasps
                    # will also be below threshold — no need to keep iterating.
                    break

                if not (self.gripper_width_min <= width <= self.gripper_width_max):
                    continue

                # BUG-FIX-H: _transform_pos/_transform_rot must not be called
                # when transform is invalid — see _transform_pos docstring.
                pos_base = self._transform_pos(pos_cam)
                rot_base = self._transform_rot(rot_cam)

                if not self._in_workspace(pos_base):
                    continue

                # Approach direction = third column of rotation matrix.
                # VERIFY THIS against your AnyGrasp SDK version before deployment.
                # Some SDK versions use column 0. Check anygrasp_sdk/README.md.
                approach_cam = rot_cam[:, 2]

                # BUG-FIX-I: approach vector must be a unit vector.
                # Rotation matrices are theoretically orthonormal but floating point
                # accumulation in the SDK can produce columns with norm != 1.0.
                # Normalise defensively before storing.
                approach_norm = np.linalg.norm(approach_cam)
                if approach_norm < 1e-6:
                    continue   # degenerate rotation matrix — skip this grasp
                approach_cam = approach_cam / approach_norm

                pre_grasp_cam  = pos_cam - approach_cam * self.pre_grasp_offset
                pre_grasp_base = self._transform_pos(pre_grasp_cam)

                grasps.append({
                    'position_cam':   pos_cam,
                    'rotation_cam':   rot_cam,
                    'width':          width,
                    'score':          score,
                    'approach':       approach_cam,
                    'source':         'anygrasp',
                    'position_base':  pos_base,
                    'rotation_base':  rot_base,
                    'pre_grasp_base': pre_grasp_base,
                })

            return grasps

        except Exception as e:
            print(f"AnyGrasp inference error: {e}")
            return []

    def _transform_pos(self, pos_cam):
        """
        Transform 3D position from camera frame to robot base frame.

        BUG-FIX-A: Raises RuntimeError if calibration was never loaded,
        instead of silently returning camera-frame coordinates labeled as
        base-frame coordinates, which would send the arm to the wrong position.
        """
        if not self._transform_valid:
            raise RuntimeError(
                "_transform_pos called but T_cam_to_base is not valid. "
                "Run hand_eye_calibration.py first."
            )
        pos_h = np.append(pos_cam, 1.0)
        return (self.T_cam_to_base @ pos_h)[:3]

    def _transform_rot(self, rot_cam):
        """
        Transform rotation matrix from camera frame to robot base frame.

        BUG-FIX-A: Same guard as _transform_pos.
        """
        if not self._transform_valid:
            raise RuntimeError(
                "_transform_rot called but T_cam_to_base is not valid. "
                "Run hand_eye_calibration.py first."
            )
        return self.T_cam_to_base[:3, :3] @ rot_cam

    def _in_workspace(self, pos_base):
        """
        Check if position is within arm's reachable workspace.

        Uses spherical shell check (min/max reach radius) which better
        approximates real arm workspace than an axis-aligned box.
        MoveIt2 does the full IK/collision check — this is a fast pre-filter.

        pos_base is in robot base frame (REP-103: +X forward, +Y left, +Z up).
        """
        reach = float(np.linalg.norm(pos_base))
        if reach < self.workspace_min_reach or reach > self.workspace_max_reach:
            return False
        if pos_base[2] < self.workspace_min_z or pos_base[2] > self.workspace_max_z:
            return False
        return True

    def visualize_grasps(self, frame, grasps, camera_matrix):
        """
        Project grasp positions onto image for visualization.

        Args:
            frame:         BGR image
            grasps:        list of grasp dicts from plan()
            camera_matrix: 3x3 K matrix

        Returns:
            frame with grasp centers overlaid (green=best, red=worst)
        """
        out = frame.copy()
        fx  = camera_matrix[0, 0]
        fy  = camera_matrix[1, 1]
        cx  = camera_matrix[0, 2]
        cy  = camera_matrix[1, 2]

        # BUG-FIX-J: Original loop used enumerate but grasps[:5] creates a new
        # list — if grasps has fewer than 5 items the color gradient is wrong
        # because t = i / max(len(grasps[:5]) - 1, 1) uses 5 not actual count.
        display_grasps = grasps[:5]
        n = len(display_grasps)

        for i, g in enumerate(display_grasps):
            pos   = g['position_cam']
            score = g['score']

            if pos[2] <= 0:
                continue

            u = int(pos[0] * fx / pos[2] + cx)
            v = int(pos[1] * fy / pos[2] + cy)

            # BUG-FIX-K: No bounds check before drawing — if the projected
            # grasp position is outside the image (e.g. a grasp on the edge of
            # the point cloud), cv2.circle and putText will clip silently on
            # some builds and raise on others. Guard explicitly.
            h, w = out.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                continue

            # Color: green = best rank, red = worst rank
            t     = i / max(n - 1, 1)
            color = (int(255 * t), int(255 * (1 - t)), 0)

            cv2.circle(out, (u, v), 8, color, -1)
            cv2.putText(
                out,
                f"G{i} s={score:.2f}",
                (u + 10, v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        return out


class TopDownGraspPlanner:
    """
    Simple fallback grasp planner when AnyGrasp is not available.
    Approaches from directly above the object centroid.

    Adequate for:
        - Flat objects on a table (bowls, books, remotes)
        - Testing the full pipeline before AnyGrasp license arrives
        - Upright objects in known orientation

    NOT adequate for:
        - Side grasp needed (tall bottles, mugs by handle)
        - Complex orientations or cluttered scenes

    WARNING: This planner assumes camera Y-axis is roughly vertical
    (pointing down). If the camera is tilted, mounted sideways, or on
    a wrist, pre_grasp[1] -= offset does NOT mean "above the object".
    Fix: compute approach direction in base frame using T_cam_to_base.

    Output matches AnyGraspPlanner dict structure exactly.
    """

    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg  = cfg
        grasp_cfg = cfg.get('anygrasp', {})

        # Gripper physical constraints
        self.gripper_width_min = grasp_cfg.get('gripper_width_min', 0.0)
        self.gripper_width_max = grasp_cfg.get('gripper_width_max', 0.10)

        # Score threshold — reject low-confidence grasps
        self.score_threshold = grasp_cfg.get('score_threshold', 0.3)

        # Spherical workspace bounds
        self.workspace_min_reach = grasp_cfg.get('workspace_min_reach_m', 0.10)
        self.workspace_max_reach = grasp_cfg.get('workspace_max_reach_m', 0.70)
        self.workspace_min_z     = grasp_cfg.get('workspace_min_z_m',     0.00)
        self.workspace_max_z     = grasp_cfg.get('workspace_max_z_m',     1.20)

        self.pre_grasp_offset = grasp_cfg.get('pre_grasp_offset_m', 0.15)

        # BUG-FIX-A: T_cam_to_base identity fallback silently poisons all
        # base-frame outputs. Added _transform_valid flag so transform methods
        # raise a clear RuntimeError instead of returning wrong values.
        self._transform_valid = False

        try:
            self.T_cam_to_base = np.load(
                cfg['transforms']['T_cam_to_base_file']
            ).astype(np.float64)

            self._transform_valid = True

        except FileNotFoundError:
            self.T_cam_to_base = np.eye(4)

            print(
                "WARNING: T_cam_to_base not found. "
                "Run hand-eye calibration first. "
                "Base-frame outputs will be invalid until calibration is loaded."
            )

        except KeyError:
            self.T_cam_to_base = np.eye(4)

            print(
                "WARNING: 'transforms.T_cam_to_base_file' missing from config. "
                "Add it to configs/config.yaml."
            )

    def plan(self, object_pos_3d, top_k=3, object_angle=None):
        """
        Generate top-down grasp candidates with small rotational variants.

        Args:
            object_pos_3d: [X,Y,Z] object centroid in camera frame
            top_k:         max candidates to return
            object_angle:  preferred gripper angle from PCA (radians), or None

        Returns:
            list of grasp dicts (same structure as AnyGraspPlanner output)
        """
        # BUG-FIX-N: No validation of object_pos_3d input. A None or wrong-shape
        # array would crash with a confusing numpy error inside the loop.
        if object_pos_3d is None:
            print("TopDownGraspPlanner.plan(): object_pos_3d is None.")
            return []

        pos = np.array(object_pos_3d, dtype=np.float64)

        if pos.shape != (3,):
            print(f"TopDownGraspPlanner.plan(): expected shape (3,), "
                  f"got {pos.shape}.")
            return []

        # BUG-FIX-O: If Z <= 0, the object is behind or at the camera plane.
        # Pre-grasp and grasp positions would be nonsensical. Reject early.
        if pos[2] <= 0:
            print(f"TopDownGraspPlanner.plan(): object Z = {pos[2]:.3f} <= 0. "
                  "Object is behind the camera. Check depth pipeline.")
            return []

        grasps = []

        # Rotational variants around approach axis
        angles = [0.0, np.pi / 6, -np.pi / 6, np.pi / 4, -np.pi / 4]
        if object_angle is not None:
            angles = [object_angle] + angles

        for angle in angles[:top_k]:
            ca, sa = np.cos(angle), np.sin(angle)

            # Rotation matrix in camera frame.
            # Approach direction = +Y (down in camera frame, assuming Y ~ vertical).
            # WARNING: assumes camera Y ≈ world down — see class docstring.
            x_axis  = np.array([ ca, 0.0, sa])
            y_axis  = np.array([ 0.0, 1.0, 0.0])   # approach direction (down)
            z_axis  = np.array([-sa, 0.0, ca])

            # BUG-FIX-P: These three column vectors are orthonormal only when
            # sa^2 + ca^2 = 1, which is guaranteed for a scalar angle. But
            # x_axis and z_axis are not always unit length — for angle=0:
            # x=[1,0,0], z=[0,0,1], y=[0,1,0] — fine.
            # For angle=pi/6: x=[sqrt(3)/2, 0, 0.5], z=[-0.5, 0, sqrt(3)/2] — fine.
            # All magnitudes are 1.0 by trig identity. This is correct. No fix needed,
            # but verify column_stack ordering matches your gripper convention.
            rot_cam = np.column_stack([x_axis, y_axis, z_axis])

            # Pre-grasp: offset backward along approach direction (−Y = up in cam)
            pre_grasp_cam    = pos.copy()
            pre_grasp_cam[1] -= self.pre_grasp_offset

            # Grasp: just above object centroid
            grasp_pos_cam    = pos.copy()
            grasp_pos_cam[1] -= 0.01   # 1 cm above centroid

            # BUG-FIX-Q: _transform_* used to silently return camera-frame
            # coordinates when T_cam_to_base = eye(4). Now guarded by
            # _transform_valid. When invalid, we still populate the dict but
            # mark base-frame fields as None so callers can detect the problem
            # rather than silently receiving wrong coordinates.
            if self._transform_valid:
                pos_base       = (self.T_cam_to_base @
                                  np.append(grasp_pos_cam, 1.0))[:3]
                rot_base       = self.T_cam_to_base[:3, :3] @ rot_cam
                pre_grasp_base = (self.T_cam_to_base @
                                  np.append(pre_grasp_cam, 1.0))[:3]
            else:
                pos_base       = None
                rot_base       = None
                pre_grasp_base = None

            grasps.append({
                'position_cam':   grasp_pos_cam,
                'rotation_cam':   rot_cam,
                'width':          0.06,
                'score':          1.0 - abs(angle) / np.pi,
                'approach':       y_axis,
                'source':         'fallback',
                'position_base':  pos_base,
                'rotation_base':  rot_base,
                'pre_grasp_base': pre_grasp_base,
            })

        return sorted(grasps, key=lambda g: g['score'], reverse=True)