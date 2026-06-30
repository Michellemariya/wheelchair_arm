# simulation/pybullet_sim.py
import pybullet as p
import pybullet_data
import numpy as np
import time
import yaml


class ArmSimulation:
    """
    PyBullet simulation for testing grasp pipeline logic.
    Replaces the real arm with a simulated one — all other pipeline
    stages (detection, localisation, planning) run identically.

    PyBullet world frame: +X forward, +Y left, +Z up (REP-103).
    This differs from the OpenCV camera frame used in the perception
    pipeline. Pass object positions through T_cam_to_base before
    calling compute_ik().

    Fixes applied vs original:
      [BUG1] Fixed-joint indexing trap: _cache_joint_limits now saves
             self.actuated_joint_indices; move_to() and start-state reads
             use those indices instead of sequential range().
      [BUG2] physicsClientId=self.client passed to every p.* call so
             the class works correctly in multi-sim environments.
      [BUG3] Interpolation loop changed to range(steps+1) so t reaches
             exactly 1.0 and the arm actually arrives at its target.
      [BUG4] move_to() reads start angles from actuated_joint_indices
             not from sequential joint indices (same trap as BUG1).
      [BUG5] compute_ik() default ee_link now uses last actuated joint
             index instead of getNumJoints()-1 (which may be fixed).
      [BUG6] disconnect() clears arm_id / object_ids to prevent stale
             ID use after the physics server is gone.
      [BUG7] __init__ wraps post-connect setup in try/except so the
             physics server is always disconnected on init failure.
    """

    def __init__(self, urdf_path=None, gui=True,
                 ee_link_index=None, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        sim_cfg = cfg.get('simulation', {})
        self.sim_rate    = sim_cfg.get('rate_hz', 240)
        self.joint_force = sim_cfg.get('joint_force', 100)

        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)       # [BUG2] store client ID
        self.ee_link_index = ee_link_index

        # [BUG7] Wrap everything after connect() in try/except so the
        #        physics server is always cleaned up on init failure.
        try:
            self._setup(urdf_path)
        except Exception:
            p.disconnect(self.client)
            raise

    def _setup(self, urdf_path):
        """Post-connect initialisation (separated so BUG7 catch is clean)."""
        # [BUG2] Every p.* call receives physicsClientId=self.client
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self.plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self.client
        )

        self.object_ids = []

        if urdf_path:
            self.arm_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.client
            )
            self._cache_joint_limits()
        else:
            self.arm_id              = None
            self.lower_limits        = []
            self.upper_limits        = []
            self.joint_ranges        = []
            self.rest_poses          = []
            self.actuated_joint_indices = []   # [BUG1] initialise here too
            print("No URDF provided — using placeholder arm")

    # ------------------------------------------------------------------
    # Joint limit caching
    # ------------------------------------------------------------------

    def _cache_joint_limits(self):
        """
        Extract joint limits from the loaded URDF and cache them.
        Also stores self.actuated_joint_indices — the PyBullet joint
        indices that correspond to movable (non-fixed) joints.

        [BUG1] This list is what move_to() and compute_ik() must use
        to avoid the fixed-joint indexing trap.

        To find your ee_link_index, read the joint names printed here
        and identify the last joint before your end-effector.
        """
        self.lower_limits           = []
        self.upper_limits           = []
        self.joint_ranges           = []
        self.rest_poses             = []
        self.actuated_joint_indices = []   # [BUG1] NEW — actual joint indices

        n_total = p.getNumJoints(
            self.arm_id, physicsClientId=self.client
        )

        print("=== URDF Joint Info ===")
        for i in range(n_total):
            info       = p.getJointInfo(
                self.arm_id, i, physicsClientId=self.client
            )
            joint_type = info[2]
            joint_name = info[12].decode('utf-8')
            print(f"  Index {i}: {joint_name} (type={joint_type})")

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                lower = info[8]
                upper = info[9]

                # Guard against zero-range joints that could confuse the IK solver
                if upper <= lower:
                    print(f"  WARNING: joint {joint_name} has zero/inverted range "
                          f"({lower:.3f}, {upper:.3f}) — skipping")
                    continue

                self.lower_limits.append(lower)
                self.upper_limits.append(upper)
                self.joint_ranges.append(upper - lower)
                self.rest_poses.append((lower + upper) / 2.0)
                self.actuated_joint_indices.append(i)   # [BUG1] store real index

        print(f"Found {len(self.actuated_joint_indices)} actuated joints: "
              f"{self.actuated_joint_indices}")
        print("======================")

    # ------------------------------------------------------------------
    # Scene objects
    # ------------------------------------------------------------------

    def add_object(self, position, shape='cylinder'):
        """Add a graspable test object to the scene."""
        if shape == 'cylinder':
            col_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.03, height=0.1,
                physicsClientId=self.client
            )
            vis_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.03, length=0.1,
                rgbaColor=[0.8, 0.3, 0.3, 1],
                physicsClientId=self.client
            )
        elif shape == 'box':
            col_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.03, 0.05, 0.04],
                physicsClientId=self.client
            )
            vis_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.03, 0.05, 0.04],
                rgbaColor=[0.3, 0.8, 0.3, 1],
                physicsClientId=self.client
            )
        else:   # sphere
            col_id = p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.04,
                physicsClientId=self.client
            )
            vis_id = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.04,
                rgbaColor=[0.3, 0.3, 0.8, 1],
                physicsClientId=self.client
            )

        obj_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            physicsClientId=self.client
        )

        self.object_ids.append(obj_id)
        return obj_id

    # ------------------------------------------------------------------
    # IK and motion
    # ------------------------------------------------------------------

    def compute_ik(self, target_position, target_orientation=None):
        """
        Compute joint angles for a target end-effector pose.
        Uses cached joint limits from URDF to constrain the solution.

        Returns:
            np.ndarray of length len(actuated_joint_indices)
        """
        if self.arm_id is None:
            return np.zeros(len(self.actuated_joint_indices))

        # [BUG5] Default to last ACTUATED joint, not last joint overall.
        #        getNumJoints()-1 is often a fixed EE-frame marker joint.
        if self.ee_link_index is not None:
            ee_link = self.ee_link_index
        else:
            if self.actuated_joint_indices:
                ee_link = self.actuated_joint_indices[-1]
            else:
                ee_link = p.getNumJoints(
                    self.arm_id, physicsClientId=self.client
                ) - 1
            print(
                "WARNING: ee_link_index not set — defaulting to last "
                "actuated joint. Check printed joint names and set "
                "ee_link_index explicitly in __init__ for correct IK."
            )

        if target_orientation is None:
            target_orientation = p.getQuaternionFromEuler([0, -np.pi / 2, 0])

        joint_angles = p.calculateInverseKinematics(
            self.arm_id,
            ee_link,
            target_position,
            target_orientation,
            lowerLimits  = self.lower_limits,
            upperLimits  = self.upper_limits,
            jointRanges  = self.joint_ranges,
            restPoses    = self.rest_poses,
            maxNumIterations   = 200,
            residualThreshold  = 1e-4,
            physicsClientId    = self.client    # [BUG2]
        )

        # Slice to the number of actuated joints we care about — IK may
        # return angles for gripper joints too if they exist in the URDF.
        n_act = len(self.actuated_joint_indices)
        return np.array(joint_angles[:n_act])

    def move_to(self, joint_angles, steps=100):
        """
        Simulate smooth arm movement by linearly interpolating from
        current joint positions to target over `steps` simulation steps.

        Args:
            joint_angles: iterable of target angles for actuated joints
            steps:        number of interpolation steps
        """
        if self.arm_id is None:
            return

        n_act  = len(self.actuated_joint_indices)
        angles = np.array(joint_angles[:n_act], dtype=float)

        # [BUG1, BUG4] Read current positions using actuated_joint_indices,
        #              NOT sequential range(). Fixed joints return 0.0 from
        #              getJointState and would corrupt the interpolation start.
        start_angles = []
        for idx in self.actuated_joint_indices:
            state = p.getJointState(
                self.arm_id, idx, physicsClientId=self.client
            )
            start_angles.append(state[0])   # index 0 = position

        # [BUG3] range(steps+1) so t reaches exactly 1.0 on the final step.
        #        Original range(steps) stopped at t=(steps-1)/steps ≈ 0.99.
        for step in range(steps + 1):
            t = step / steps    # 0.0 → 1.0  (inclusive on both ends)

            for j, idx in enumerate(self.actuated_joint_indices):
                interp = start_angles[j] + t * (angles[j] - start_angles[j])
                p.setJointMotorControl2(
                    self.arm_id, idx,           # [BUG1] use real index, not j
                    controlMode    = p.POSITION_CONTROL,
                    targetPosition = interp,
                    force          = self.joint_force,
                    physicsClientId = self.client   # [BUG2]
                )

            p.stepSimulation(physicsClientId=self.client)   # [BUG2]
            time.sleep(1.0 / self.sim_rate)

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self):
        """Advance the simulation by one physics step."""
        p.stepSimulation(physicsClientId=self.client)   # [BUG2]

    def disconnect(self):
        """
        Shut down the physics server and clear all stale body IDs.
        [BUG6] Clears arm_id and object_ids so any accidental post-
        disconnect method calls fail fast at the arm_id=None guards
        instead of crashing PyBullet with cryptic stale-ID errors.
        """
        p.disconnect(self.client)
        self.arm_id     = None      # [BUG6]
        self.object_ids = []        # [BUG6]