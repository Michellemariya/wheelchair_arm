# planning/grasp/grasp_planner.py
import numpy as np
import yaml


class GraspPlanner:
    """
    Computes gripper target poses given a 3D object position and orientation.

    Supports:
        - Top-down grasp:         gripper approaches vertically from above
        - Aligned top-down grasp: same, but fingers aligned to object principal axis
        - Side grasp:             gripper approaches horizontally along principal axis

    All poses are first computed in camera frame, then transformed to arm base
    frame via T_cam_to_base (requires completed hand-eye calibration).

    COORDINATE FRAME ASSUMPTION
    ---------------------------
    This planner assumes the camera is mounted such that its Y axis points
    roughly downward (toward the floor / tabletop). Z points forward into the
    scene. If your camera is tilted, pass an explicit `approach_dir` to the
    top-down methods. This assumption is NEVER silently enforced — it is
    checked only by geometry, not by code.

    Fixes applied vs original:
      [BUG-A] prefer_top_down parameter now actually controls grasp ordering
      [BUG-B] Zero-length axis guard added to compute_side_grasp
      [BUG-C] Double-degenerate fallback replaced with robust cardinal-axis picker
      [BUG-D] T_cam_to_base fallback raises RuntimeError in transform methods
              instead of silently returning camera-frame values as base-frame
      [BUG-E] Grasp depth (was hardcoded 0.01 m) is now a configurable parameter
      [BUG-F] Unreachable None guard replaced with explicit docstring contract
      [BUG-G] Alignment rotation expressed as right-multiply in gripper local frame
      [BUG-H] Top-down methods accept explicit approach_dir parameter
    """

    # Default approach direction: camera Y points down toward tabletop
    _DEFAULT_APPROACH = np.array([0.0, 1.0, 0.0])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        with open(cfg['camera']['calibration_file']) as f:
            calib_data = yaml.safe_load(f)
        self.K = np.array(calib_data['camera_matrix']['data'],
                          dtype=np.float64).reshape(3, 3)

        # [BUG-D] Track whether the transform is actually valid.
        #         Transform methods raise RuntimeError if this is False,
        #         instead of silently returning camera-frame values.
        self._transform_valid = False

        try:
            self.T_cam_to_base = np.load(
                cfg['transforms']['T_cam_to_base_file']
            ).astype(np.float64)
            self._transform_valid = True
        except FileNotFoundError:
            print(
                "[GraspPlanner] WARNING: T_cam_to_base not found. "
                "Run hand-eye calibration first. "
                "Calls to transform_pose_to_base() will raise RuntimeError "
                "until calibration is loaded via load_transform()."
            )
            self.T_cam_to_base = np.eye(4)

        # [BUG-E] Load grasp depth from config (default 0.01 m)
        self._default_grasp_depth = float(
            cfg.get('grasp', {}).get('grasp_depth_m', 0.01)
        )

    def load_transform(self, T_cam_to_base: np.ndarray):
        """
        Manually provide or update the camera-to-base transform.
        Useful when calibration is loaded after construction.
        """
        self.T_cam_to_base = np.array(T_cam_to_base, dtype=np.float64)
        self._transform_valid = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_normalize(v: np.ndarray, name: str = 'vector') -> np.ndarray:
        """
        Normalize a vector; raise ValueError with a descriptive message
        if it is zero-length, rather than silently producing NaN.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            raise ValueError(
                f"[GraspPlanner] Cannot normalize {name}: "
                f"zero-length vector {v}. "
                f"Check that your object axis / approach inputs are valid."
            )
        return v / norm

    @staticmethod
    def _perpendicular_to(v: np.ndarray) -> np.ndarray:
        """
        Return any unit vector perpendicular to v.

        [BUG-C] Robust implementation: picks the cardinal axis that gives
        the largest cross-product magnitude (avoids the double-degenerate
        fallback that could produce a second zero vector).
        """
        v = v / np.linalg.norm(v)
        candidates = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        best = max(candidates, key=lambda c: np.linalg.norm(np.cross(v, c)))
        return GraspPlanner._safe_normalize(np.cross(v, best), 'perpendicular')

    @staticmethod
    def _ensure_orthogonal(R: np.ndarray) -> np.ndarray:
        """
        Re-orthogonalise a 3x3 rotation matrix via SVD to remove accumulated
        floating-point error from repeated matrix multiplications.
        Returns a proper rotation with det = +1.
        """
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt
        return R_clean

    @staticmethod
    def _build_rotation(approach_dir: np.ndarray) -> np.ndarray:
        """
        Build a gripper rotation matrix given the desired approach axis
        (which becomes the gripper's Z column).

        X column = finger-spread direction (perpendicular to approach,
                   lying as close to horizontal as possible)
        Y column = cross(Z, X)  (completes right-hand frame)
        Z column = approach_dir
        """
        z_axis = GraspPlanner._safe_normalize(approach_dir, 'approach_dir')
        # Choose finger-spread direction: prefer camera X axis [1,0,0];
        # fall back robustly if parallel.
        x_axis = GraspPlanner._perpendicular_to(z_axis)
        y_axis = GraspPlanner._safe_normalize(np.cross(z_axis, x_axis), 'y_axis')
        return np.column_stack([x_axis, y_axis, z_axis])

    # ------------------------------------------------------------------
    # Grasp computation
    # ------------------------------------------------------------------

    def compute_top_down_grasp(self, object_pos_3d,
                                pre_grasp_offset: float = 0.15,
                                grasp_depth: float = None,
                                approach_dir: np.ndarray = None):
        """
        Approach from directly above the object.

        Works for most tabletop objects.

        Args:
            object_pos_3d:    [X,Y,Z] object centroid in camera frame (m)
            pre_grasp_offset: distance above object to start approach (m)
            grasp_depth:      how far above object centre to stop and close
                              gripper (m).  Defaults to config value (0.01 m).
            approach_dir:     unit vector pointing toward the object in camera
                              frame.  Defaults to [0,1,0] (camera Y down).
                              [BUG-H] Override this if your camera is tilted.

        Returns:
            dict with 'type', 'pre_grasp', 'grasp', 'R', 'approach_vec'
            (all in camera frame)
        """
        pos  = np.array(object_pos_3d, dtype=np.float64)
        gd   = grasp_depth if grasp_depth is not None else self._default_grasp_depth
        adir = self._safe_normalize(
            np.array(approach_dir if approach_dir is not None
                     else self._DEFAULT_APPROACH, dtype=np.float64),
            'approach_dir'
        )

        pre_grasp = pos - adir * pre_grasp_offset   # back off along approach axis
        grasp     = pos - adir * gd                 # [BUG-E] was hardcoded 0.01

        R_grasp = self._build_rotation(adir)
        R_grasp = self._ensure_orthogonal(R_grasp)

        return {
            'type':         'top_down',
            'pre_grasp':    pre_grasp,
            'grasp':        grasp,
            'R':            R_grasp,
            'approach_vec': adir.copy()             # direction toward object
        }

    def compute_aligned_top_down_grasp(self, object_pos_3d,
                                        object_angle,
                                        pre_grasp_offset: float = 0.15,
                                        grasp_depth: float = None,
                                        approach_dir: np.ndarray = None):
        """
        Top-down grasp with gripper fingers aligned to object principal axis.
        Better for elongated objects (pens, bottles, TV remotes).

        Args:
            object_angle: principal axis angle from mask PCA (radians),
                          measured from positive image x-axis.
        """
        pose = self.compute_top_down_grasp(
            object_pos_3d, pre_grasp_offset, grasp_depth, approach_dir
        )

        if object_angle is None:
            return pose

        # [BUG-G] Original code left-multiplied R_align (rotation in world/camera
        #         frame). This is coincidentally correct only because the approach
        #         axis happens to be [0,1,0] = world Y.  The invariant intent is:
        #         "rotate the gripper around its OWN approach/Z axis".
        #         That is always expressed as a RIGHT-multiply: R_new = R @ R_local.
        cos_a = np.cos(object_angle)
        sin_a = np.sin(object_angle)

        # Rotation around local Z axis (gripper approach axis)
        R_align_local = np.array([
            [ cos_a, -sin_a, 0.0],
            [ sin_a,  cos_a, 0.0],
            [  0.0,    0.0,  1.0]
        ])

        pose['R']    = self._ensure_orthogonal(pose['R'] @ R_align_local)  # [BUG-G]
        pose['type'] = 'aligned_top_down'
        return pose

    def compute_side_grasp(self, object_pos_3d, object_axis_3d,
                            pre_grasp_offset: float = 0.15,
                            grasp_depth: float = None,
                            up_dir: np.ndarray = None):
        """
        Approach from the side along the principal axis.
        Better for tall objects (upright bottles, cans).

        Args:
            object_pos_3d:    [X,Y,Z] object centroid in camera frame (m)
            object_axis_3d:   principal 3D axis of the object in camera frame
            pre_grasp_offset: standoff distance before approach (m)
            grasp_depth:      final depth into approach (m)
            up_dir:           "up" direction in camera frame.
                              Defaults to [0,-1,0] (camera -Y = upward).
        """
        pos  = np.array(object_pos_3d, dtype=np.float64)
        gd   = grasp_depth if grasp_depth is not None else self._default_grasp_depth

        # [BUG-B] Guard against zero-length axis input
        axis_raw = np.array(object_axis_3d, dtype=np.float64)
        axis = self._safe_normalize(axis_raw, 'object_axis_3d')

        up = self._safe_normalize(
            np.array(up_dir if up_dir is not None
                     else [0.0, -1.0, 0.0], dtype=np.float64),
            'up_dir'
        )

        # Approach direction: perpendicular to principal axis, horizontal
        # [BUG-C] Replaced naive two-candidate fallback with robust cardinal picker
        raw_approach = np.cross(axis, up)
        if np.linalg.norm(raw_approach) < 1e-6:
            # axis is parallel to up — use robust perpendicular finder
            raw_approach = self._perpendicular_to(axis)

        approach = self._safe_normalize(raw_approach, 'approach')

        pre_grasp = pos + approach * pre_grasp_offset
        grasp     = pos + approach * gd

        # Build rotation: gripper Z points toward object (-approach)
        z_axis = self._safe_normalize(-approach, '-approach')
        y_axis = up
        x_axis = self._safe_normalize(np.cross(y_axis, z_axis), 'x_axis')
        # Re-orthogonalise y after x is fixed
        y_axis = self._safe_normalize(np.cross(z_axis, x_axis), 'y_axis')

        R_grasp = self._ensure_orthogonal(np.column_stack([x_axis, y_axis, z_axis]))

        return {
            'type':         'side',
            'pre_grasp':    pre_grasp,
            'grasp':        grasp,
            'R':            R_grasp,
            'approach_vec': z_axis.copy()       # points toward object
        }

    # ------------------------------------------------------------------
    # Frame transforms
    # ------------------------------------------------------------------

    def _check_transform(self):
        """
        [BUG-D] Raise early with a clear message rather than silently
        returning camera-frame values labeled as base-frame values.
        """
        if not self._transform_valid:
            raise RuntimeError(
                "[GraspPlanner] T_cam_to_base is not loaded. "
                "Run hand-eye calibration and call load_transform(), "
                "or provide 'T_cam_to_base_file' in your config before "
                "calling any transform_*_to_base() method."
            )

    def transform_pose_to_base(self, pos_camera_frame: np.ndarray) -> np.ndarray:
        """
        Transform a 3D position from camera frame to arm base frame.
        Raises RuntimeError if hand-eye calibration has not been loaded.
        """
        self._check_transform()         # [BUG-D]
        pos_h = np.append(pos_camera_frame, 1.0)
        return (self.T_cam_to_base @ pos_h)[:3]

    def transform_rotation_to_base(self, R_camera_frame: np.ndarray) -> np.ndarray:
        """
        Transform a rotation matrix from camera frame to arm base frame.
        Raises RuntimeError if hand-eye calibration has not been loaded.
        """
        self._check_transform()         # [BUG-D]
        R_cam_to_base = self.T_cam_to_base[:3, :3]
        return self._ensure_orthogonal(R_cam_to_base @ R_camera_frame)

    # ------------------------------------------------------------------
    # Grasp selection
    # ------------------------------------------------------------------

    def select_best_grasp(self, object_pos_3d, object_angle,
                           object_axis_3d=None,
                           prefer_top_down: bool = True,
                           approach_dir: np.ndarray = None):
        """
        Select the most appropriate grasp type given object properties.

        Args:
            prefer_top_down: if True, prioritises aligned_top_down → top_down
                             → side.  If False, prioritises side → top_down.
                             [BUG-A] This parameter now actually controls ordering.

        Returns:
            List of grasp dicts sorted by preference, each with added keys:
                pre_grasp_base, grasp_base, R_base
            The first element is the recommended grasp.
            List is never empty (top_down is always computable).

        Raises:
            RuntimeError if hand-eye calibration not loaded (from transform calls).
        """
        top_down_grasps = []
        side_grasps     = []

        # Always compute top-down options
        if object_angle is not None:
            top_down_grasps.append(
                self.compute_aligned_top_down_grasp(
                    object_pos_3d, object_angle,
                    approach_dir=approach_dir
                )
            )
        top_down_grasps.append(
            self.compute_top_down_grasp(
                object_pos_3d,
                approach_dir=approach_dir
            )
        )

        # Side grasp only if a 3D axis is available
        if object_axis_3d is not None:
            try:
                side_grasps.append(
                    self.compute_side_grasp(object_pos_3d, object_axis_3d)
                )
            except ValueError as e:
                # [BUG-B/C] Degenerate axis — skip side grasp gracefully
                print(f"[GraspPlanner] Skipping side grasp: {e}")

        # [BUG-A] Actually honour prefer_top_down
        if prefer_top_down:
            grasps = top_down_grasps + side_grasps
        else:
            grasps = side_grasps + top_down_grasps

        # Transform all candidates to base frame
        for g in grasps:
            g['pre_grasp_base'] = self.transform_pose_to_base(g['pre_grasp'])
            g['grasp_base']     = self.transform_pose_to_base(g['grasp'])
            g['R_base']         = self.transform_rotation_to_base(g['R'])

        # [BUG-F] grasps is guaranteed non-empty (top_down always added above).
        #         The old 'if grasps else None' guard was dead code that gave
        #         false confidence. Returning the full ranked list instead of
        #         just grasps[0] lets callers implement reachability filtering.
        return grasps