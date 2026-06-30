# planning/motion/motion_planner.py
#
# Motion planning layer.
#
# TWO BACKENDS:
#   1. PyBullet  — for simulation and development (available now)
#   2. MoveIt2   — for real hardware execution (requires URDF + ROS2)
#
# HOW TO USE:
#   During development (no real arm, no URDF):
#       planner = MotionPlanner(backend='pybullet')
#
#   When URDF is ready and ROS2 is set up:
#       planner = MotionPlanner(backend='moveit2')
#
#   Auto (use MoveIt2 if available, else PyBullet):
#       planner = MotionPlanner(backend='auto')
#
# MOVEIT2 SETUP (do this when mechanical team provides URDF):
# ============================================================
# 1. Install MoveIt2:
#       sudo apt install ros-humble-moveit
#
# 2. Run MoveIt Setup Assistant with your arm's URDF:
#       ros2 launch moveit_setup_assistant setup_assistant.launch.py
#       - Load URDF
#       - Define planning group "arm" (all joints)
#       - Define end effector group "gripper"
#       - Auto-generate collision matrix (SRDF)
#       - Configure kinematics solver (recommend: KDL or TracIK)
#           TracIK is faster and handles singularities better
#           sudo apt install ros-humble-trac-ik-kinematics-plugin
#       - Generate and save config package
#
# 3. Launch MoveIt2:
#       ros2 launch your_arm_moveit_config move_group.launch.py
#
# 4. Test in RViz:
#       ros2 launch your_arm_moveit_config moveit_rviz.launch.py
#
# MOVEIT2 RESOURCES:
#   Official docs:   https://moveit.picknik.ai/
#   Python API:      https://github.com/moveit/moveit2/tree/main/moveit_py
#   Tutorial:        https://moveit.picknik.ai/main/doc/tutorials/tutorials.html
#   TracIK plugin:   https://ros-planning.github.io/moveit_tutorials/
#
# PYBULLET RESOURCES:
#   Quickstart:      https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
#   Examples:        https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/examples
#   Robot URDFs:     https://github.com/bulletphysics/bullet3/tree/master/data
#
# IMPORTANT — URDF SOURCE:
#   Get the URDF from the mechanical team.
#   It must include: link geometry, joint limits, DH parameters.
#   If no URDF yet, use a placeholder 4-DOF arm from PyBullet data.
#
# IMPORTANT — COORDINATE FRAME:
#   All position inputs to this module must be in ROBOT BASE FRAME.
#   Call GraspPlanner.transform_pose_to_base() on any perception output
#   (tvec, pixel_to_3d, etc.) BEFORE passing it here.
#   Passing camera-frame coordinates will silently move arm to wrong position.

import numpy as np
import yaml
import time


# =====================================================================
# PyBullet Backend
# =====================================================================

class PyBulletPlanner:
    """
    Motion planning using PyBullet for simulation.

    Use this during development before MoveIt2 is configured.
    All other pipeline stages (perception, grasp planning) run identically.
    Only this module changes when switching to real hardware.
    """

    def __init__(self, urdf_path=None, config_path='configs/config.yaml',
                 gui=True):
        import pybullet as p
        import pybullet_data

        self.p = p
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground
        p.loadURDF("plane.urdf")

        # Load config
        cfg = {}
        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                print(f"Config not found at {config_path}, using defaults")

        sim_cfg = cfg.get('simulation', {})

        # Fix #12: read joint_force from config instead of hardcoding 500
        # Config already defines simulation.joint_force: 100
        self.joint_force = sim_cfg.get('joint_force', 100)

        # Load arm
        if urdf_path:
            self.arm_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True
            )
            self.n_joints = p.getNumJoints(self.arm_id)

            # Fix #1: ee_link_index from config, NOT n_joints - 1.
            # n_joints - 1 points at the last link in the URDF which may be a
            # camera mount, passive link, or gripper finger — not the tool centre point.
            # Set simulation.ee_link_index in config once you inspect the URDF with:
            #   for i in range(p.getNumJoints(arm_id)): print(i, p.getJointInfo(arm_id, i)[12])
            self.ee_link = sim_cfg.get('ee_link_index', self.n_joints - 1)
            if 'ee_link_index' not in sim_cfg:
                print(f"WARNING: ee_link_index not set in config. "
                      f"Defaulting to {self.ee_link} (n_joints-1). "
                      f"Verify this is the gripper tip, not a camera/passive link. "
                      f"Set simulation.ee_link_index in configs/config.yaml.")
            print(f"Loaded arm URDF: {urdf_path}, {self.n_joints} joints, "
                  f"ee_link={self.ee_link}")
        else:
            # Placeholder: load a simple arm from PyBullet data
            # Replace with your arm's URDF when available
            try:
                self.arm_id = p.loadURDF("kuka_iiwa/model.urdf",
                                         useFixedBase=True)
                self.n_joints = p.getNumJoints(self.arm_id)
                # Fix #1: same config-driven ee_link for placeholder too
                self.ee_link = sim_cfg.get('ee_link_index', self.n_joints - 1)
                if 'ee_link_index' not in sim_cfg:
                    print(f"WARNING: ee_link_index not set in config. "
                          f"Defaulting to {self.ee_link} (n_joints-1).")
                print("Loaded placeholder KUKA arm (replace with your URDF)")
            except Exception:
                self.arm_id   = None
                self.n_joints = 5
                self.ee_link  = 4
                print("No URDF loaded. IK will return zeros.")

        # Fix #11: filter revolute joints only — excludes fixed, mimic, passive joints.
        # plan_to_pose and execute_trajectory only operate on this list.
        self.joint_indices = []
        self.joint_limits  = {'lower': [], 'upper': []}

        if self.arm_id is not None:
            for i in range(self.n_joints):
                info = p.getJointInfo(self.arm_id, i)
                if info[2] == p.JOINT_REVOLUTE:
                    self.joint_indices.append(i)
                    self.joint_limits['lower'].append(info[8])
                    self.joint_limits['upper'].append(info[9])

    # ------------------------------------------------------------------
    # IK + FK verification
    # ------------------------------------------------------------------

    def compute_ik(self, target_pos, target_orn=None):
        """
        Compute joint angles for target end-effector pose.

        Args:
            target_pos: [X,Y,Z] in robot base frame (meters)
            target_orn: quaternion [x,y,z,w] or None (free orientation)

        Returns:
            joint_angles: numpy array of joint angles (radians), or None if IK failed.
        """
        if self.arm_id is None:
            return np.zeros(self.n_joints)

        if target_orn is None:
            target_orn = self.p.getQuaternionFromEuler([0, -np.pi/2, 0])

        angles = self.p.calculateInverseKinematics(
            self.arm_id,
            self.ee_link,
            target_pos,
            target_orn,
            lowerLimits=self.joint_limits['lower'],
            upperLimits=self.joint_limits['upper'],
            jointRanges=[u - l for u, l in zip(
                self.joint_limits['upper'],
                self.joint_limits['lower']
            )],
            restPoses=[0.0] * len(self.joint_indices),
            maxNumIterations=200,
            residualThreshold=1e-4
        )

        angles = np.array(angles[:len(self.joint_indices)])

        # Fix #3/#8: PyBullet always returns a result, even for unreachable targets.
        # Verify by running FK on the solution and checking position error.
        if not self._verify_ik(angles, target_pos):
            return None

        return angles

    def _verify_ik(self, joint_angles, target_pos, tolerance=0.01):
        """
        Verify IK solution by running FK and checking end-effector position error.

        PyBullet's calculateInverseKinematics returns its best-effort result
        even when the target is unreachable. This catches silent failures.

        Args:
            joint_angles: IK solution to verify
            target_pos:   [X,Y,Z] the IK was solved for
            tolerance:    max acceptable position error in meters (default 1cm)

        Returns:
            True if solution is valid, False if target was unreachable.
        """
        # Temporarily set joints to IK solution to read FK result
        for idx, angle in zip(self.joint_indices, joint_angles):
            self.p.resetJointState(self.arm_id, idx, angle)

        state = self.p.getLinkState(self.arm_id, self.ee_link)
        achieved_pos = np.array(state[4])   # world position of EE link
        error = np.linalg.norm(achieved_pos - np.array(target_pos))

        if error > tolerance:
            print(f"IK verification failed: "
                  f"target={np.array(target_pos).round(4)}, "
                  f"achieved={achieved_pos.round(4)}, "
                  f"error={error:.4f}m (tolerance={tolerance}m). "
                  f"Target may be outside workspace.")
            return False

        return True

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_to_pose(self, target_pos, target_orn=None):
        """
        Plan a joint-space trajectory to reach target pose.

        Returns list of waypoints (joint angle arrays), or [] on IK failure.

        NOTE: Uses linear interpolation in joint space. This is NOT
        collision-aware. Safe for simulation. Use MoveIt2 on real hardware.

        Args:
            target_pos: [X,Y,Z] in robot base frame (meters)
            target_orn: quaternion [x,y,z,w] or None
        """
        if self.arm_id is None:
            return []

        # Current joint angles
        current_angles = np.array([
            self.p.getJointState(self.arm_id, i)[0]
            for i in self.joint_indices
        ])

        # Fix #3: compute_ik now returns None on failure
        target_angles = self.compute_ik(target_pos, target_orn)
        if target_angles is None:
            print(f"IK failed for target position {target_pos} — "
                  f"target may be outside workspace.")
            return []

        # Linear interpolation in joint space (N waypoints)
        # WARNING: not collision-aware — use MoveIt2 for real hardware
        n_steps = 50
        waypoints = []
        for i in range(n_steps + 1):
            t = i / n_steps
            wp = (1 - t) * current_angles + t * target_angles
            waypoints.append(wp)

        return waypoints

    def plan_to_joint_angles(self, target_joints):
        """
        Plan directly to a joint configuration without IK.
        Used for move_home() where target is defined in joint space.

        Args:
            target_joints: array of joint angles (radians), length = n revolute joints

        Returns:
            list of waypoint arrays
        """
        if self.arm_id is None:
            return []

        current_angles = np.array([
            self.p.getJointState(self.arm_id, i)[0]
            for i in self.joint_indices
        ])

        n_joints = len(self.joint_indices)
        target = np.array(target_joints[:n_joints])

        n_steps = 50
        return [
            (1 - t / n_steps) * current_angles + (t / n_steps) * target
            for t in range(n_steps + 1)
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_trajectory(self, waypoints, dt=0.02):
        """
        Execute trajectory in simulation.

        Args:
            waypoints: list of joint angle arrays from plan_to_pose()
            dt:        time step between waypoints (seconds)
        """
        if self.arm_id is None or not waypoints:
            return

        for wp in waypoints:
            for idx, joint_idx in enumerate(self.joint_indices):
                self.p.setJointMotorControl2(
                    self.arm_id, joint_idx,
                    controlMode=self.p.POSITION_CONTROL,
                    targetPosition=float(wp[idx]),
                    force=self.joint_force   # Fix #12: from config, not hardcoded 500
                )
            self.p.stepSimulation()
            time.sleep(dt)

    def move_to(self, target_pos, target_orn=None):
        """
        Plan and execute motion to target pose.

        Args:
            target_pos: [X,Y,Z] in robot base frame (meters)
            target_orn: quaternion [x,y,z,w] or None

        Returns:
            True if motion succeeded, False if IK failed
        """
        waypoints = self.plan_to_pose(target_pos, target_orn)
        if not waypoints:
            return False

        self.execute_trajectory(waypoints)
        return True

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def get_ee_pose(self):
        """
        Get current end-effector position and orientation.

        Returns:
            pos: [X,Y,Z] in robot base frame
            R:   3x3 rotation matrix
        """
        if self.arm_id is None:
            return np.zeros(3), np.eye(3)

        state = self.p.getLinkState(self.arm_id, self.ee_link)
        pos = np.array(state[4])    # world position
        orn = np.array(state[5])    # quaternion

        R = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos, R

    # ------------------------------------------------------------------
    # Collision objects (visualisation only — planner does not use them)
    # ------------------------------------------------------------------

    def add_collision_object(self, position, size=(0.05, 0.05, 0.1),
                              shape='box'):
        """
        Add obstacle to simulation.

        WARNING: The PyBullet planner uses linear joint interpolation and does NOT
        check for collisions during planning. This object is visible in the GUI
        and will physically block the simulated arm AFTER contact, but it will
        NOT prevent the planner from routing through it.
        For collision-aware planning, switch to MoveIt2 backend.
        """
        if shape == 'box':
            col = self.p.createCollisionShape(
                self.p.GEOM_BOX, halfExtents=np.array(size)/2
            )
        elif shape == 'cylinder':
            col = self.p.createCollisionShape(
                self.p.GEOM_CYLINDER,
                radius=size[0], height=size[2]
            )
        else:
            col = self.p.createCollisionShape(
                self.p.GEOM_SPHERE, radius=size[0]
            )

        obj_id = self.p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            basePosition=position
        )
        return obj_id

    def disconnect(self):
        self.p.disconnect(self.client)


# =====================================================================
# MoveIt2 Backend
# =====================================================================

class MoveIt2Planner:
    """
    Motion planning using MoveIt2 via moveit_py.

    Prerequisites:
        1. ROS2 Humble installed and sourced
        2. MoveIt2 installed: sudo apt install ros-humble-moveit
        3. Your arm's MoveIt2 config package generated and launched
        4. moveit_py installed: pip install moveit (or from source)

    This class wraps the moveit_py Python API.
    Full docs: https://moveit.picknik.ai/main/doc/examples/moveit_py/moveit_py_tutorial.html
    """

    def __init__(self, config_path='configs/config.yaml',
                 planning_group='arm',
                 ee_frame='gripper_link'):
        self.planning_group = planning_group
        self.ee_frame = ee_frame
        self.moveit = None

        try:
            import rclpy
            from moveit.planning import MoveItPy
            from moveit.core.robot_state import RobotState

            if not rclpy.ok():
                rclpy.init()

            self.moveit = MoveItPy(node_name='grasp_motion_planner')
            self.arm    = self.moveit.get_planning_component(planning_group)
            self.robot  = self.moveit.get_robot_model()

            print(f"MoveIt2 initialized. Planning group: {planning_group}")

        except ImportError:
            print("moveit_py not found.")
            print("Install: sudo apt install ros-humble-moveit")
            print("Then rebuild your workspace.")
        except Exception as e:
            print(f"MoveIt2 init failed: {e}")
            print("Is the move_group node running?")
            print("Launch: ros2 launch your_arm_moveit_config "
                  "move_group.launch.py")

    def is_available(self):
        return self.moveit is not None

    def move_to_pose(self, target_pos, target_orn_quat,
                     frame_id='base_link',
                     velocity_scale=0.3,
                     acceleration_scale=0.3):
        """
        Plan and execute motion to target pose.

        Args:
            target_pos:           [X,Y,Z] in robot base frame (meters)
            target_orn_quat:      [x,y,z,w] quaternion (will be normalised by caller)
            frame_id:             reference frame for target pose
            velocity_scale:       0..1, fraction of max joint velocity
            acceleration_scale:   0..1

        Returns:
            True if motion succeeded
        """
        if not self.is_available():
            print("MoveIt2 not available")
            return False

        try:
            from geometry_msgs.msg import PoseStamped

            # Build target pose message
            target_pose = PoseStamped()
            target_pose.header.frame_id = frame_id
            target_pose.pose.position.x = float(target_pos[0])
            target_pose.pose.position.y = float(target_pos[1])
            target_pose.pose.position.z = float(target_pos[2])
            target_pose.pose.orientation.x = float(target_orn_quat[0])
            target_pose.pose.orientation.y = float(target_orn_quat[1])
            target_pose.pose.orientation.z = float(target_orn_quat[2])
            target_pose.pose.orientation.w = float(target_orn_quat[3])

            # Set start state to current
            self.arm.set_start_state_to_current_state()

            # Set goal
            self.arm.set_goal_state(
                pose_stamped_msg=target_pose,
                pose_link=self.ee_frame
            )

            # Plan
            plan_result = self.arm.plan(
                single_plan_parameters={
                    'max_vel_scaling_factor':  velocity_scale,
                    'max_acc_scaling_factor':  acceleration_scale,
                }
            )

            if not plan_result:
                print("MoveIt2 planning failed")
                return False

            # Execute
            robot_traj = plan_result.trajectory
            result = self.moveit.execute(robot_traj, controllers=[])

            return result

        except Exception as e:
            print(f"MoveIt2 move_to_pose error: {e}")
            return False

    def move_to_joint_angles(self, joint_angles, velocity_scale=0.3):
        """
        Move arm to specific joint configuration.

        Args:
            joint_angles:   array of joint angles (radians)
            velocity_scale: 0..1
        """
        if not self.is_available():
            return False

        try:
            from moveit.core.robot_state import RobotState

            robot_state = RobotState(self.robot)
            robot_state.set_joint_group_positions(
                self.planning_group, list(joint_angles)
            )

            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(robot_state=robot_state)

            plan = self.arm.plan()
            if plan:
                return self.moveit.execute(plan.trajectory, controllers=[])

            return False

        except Exception as e:
            print(f"MoveIt2 joint move error: {e}")
            return False

    def move_home(self, velocity_scale=0.2):
        """Move to predefined home configuration (defined in SRDF)."""
        if not self.is_available():
            return False
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(configuration_name='home')
            plan = self.arm.plan()
            if plan:
                return self.moveit.execute(plan.trajectory, controllers=[])
            return False
        except Exception as e:
            print(f"MoveIt2 home move error: {e}")
            return False

    def add_collision_box(self, name, position, dimensions,
                           frame_id='base_link'):
        """
        Add collision object to MoveIt2 planning scene.
        Prevents the planner from routing through obstacles.

        NOTE (Issue #16): The planning scene update method used here
        (direct append to world.collision_objects) may not correctly
        notify the planning scene monitor in all MoveIt2 versions.
        Test this explicitly after connecting to real hardware.
        A more robust approach uses apply_planning_scene() — update this
        once you verify the planning scene behaviour in your MoveIt2 setup.

        Args:
            name:       unique string identifier
            position:   [X,Y,Z] center of box in frame_id
            dimensions: [length, width, height]
            frame_id:   reference frame (default: base_link)
        """
        if not self.is_available():
            return

        try:
            from moveit_msgs.msg import CollisionObject
            from shape_msgs.msg import SolidPrimitive
            from geometry_msgs.msg import Pose

            co = CollisionObject()
            co.id = name
            co.header.frame_id = frame_id

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = list(dimensions)

            pose = Pose()
            pose.position.x = float(position[0])
            pose.position.y = float(position[1])
            pose.position.z = float(position[2])
            pose.orientation.w = 1.0

            co.primitives      = [box]
            co.primitive_poses = [pose]
            co.operation = CollisionObject.ADD

            self.moveit.get_planning_scene_monitor()\
                .get_planning_scene_message()\
                .world.collision_objects.append(co)

        except Exception as e:
            print(f"Add collision object error: {e}")


# =====================================================================
# Unified Interface
# =====================================================================

class MotionPlanner:
    """
    Unified motion planning interface.

    Selects PyBullet or MoveIt2 backend based on configuration
    and availability. The rest of the pipeline calls this class
    and doesn't need to know which backend is active.

    Usage:
        # Development (simulation)
        planner = MotionPlanner(backend='pybullet', urdf_path='arm.urdf')

        # Production (real arm with MoveIt2)
        planner = MotionPlanner(backend='moveit2')

        # Auto (use MoveIt2 if available, else PyBullet)
        planner = MotionPlanner(backend='auto')

    COORDINATE FRAME:
        All inputs to move_to_pose() must be in ROBOT BASE FRAME.
        Call GraspPlanner.transform_pose_to_base() on any perception output
        before passing it here. See module docstring for details.
    """

    def __init__(self, backend='auto', urdf_path=None,
                 config_path='configs/config.yaml', gui=True):

        self.backend_name = backend
        self.planner = None
        self._config_path = config_path   # Fix #6/#15: store for move_home()

        if backend == 'moveit2':
            self.planner = MoveIt2Planner(config_path)
            if not self.planner.is_available():
                print("MoveIt2 unavailable — falling back to PyBullet")
                self.planner = PyBulletPlanner(urdf_path, config_path, gui)
                self.backend_name = 'pybullet_fallback'

        elif backend == 'pybullet':
            self.planner = PyBulletPlanner(urdf_path, config_path, gui)

        elif backend == 'auto':
            moveit = MoveIt2Planner(config_path)
            if moveit.is_available():
                self.planner = moveit
                self.backend_name = 'moveit2'
                print("Using MoveIt2 backend")
            else:
                self.planner = PyBulletPlanner(urdf_path, config_path, gui)
                self.backend_name = 'pybullet'
                print("Using PyBullet backend (MoveIt2 not available)")

        print(f"Motion planner: {self.backend_name}")

    def move_to_pose(self, target_pos, target_orn=None, velocity_scale=0.3):
        """
        Move end effector to target pose.

        CRITICAL — COORDINATE FRAME:
            target_pos must be in ROBOT BASE FRAME, NOT camera frame.
            Call GraspPlanner.transform_pose_to_base() on any perception output
            (tvec, pixel_to_3d, keypoint 3D position, etc.) BEFORE passing here.
            Passing camera-frame coordinates will move the arm to the wrong
            position with no error or warning.

        Args:
            target_pos:     [X, Y, Z] in robot base frame (meters)
            target_orn:     quaternion [x, y, z, w], or None for default
                            downward-facing gripper orientation.
                            Will be normalised automatically.
            velocity_scale: 0..1 fraction of max velocity (MoveIt2 only)

        Returns:
            True if motion succeeded
        """
        # Fix #4: normalise quaternion before passing to either backend.
        # Prevents crashes or undefined behaviour from unnormalised input.
        if target_orn is not None:
            target_orn = np.array(target_orn, dtype=float)
            norm = np.linalg.norm(target_orn)
            if norm < 1e-6:
                raise ValueError(
                    "target_orn quaternion has zero norm — check your input."
                )
            target_orn = target_orn / norm

        if isinstance(self.planner, MoveIt2Planner):
            if target_orn is None:
                from scipy.spatial.transform import Rotation
                target_orn = Rotation.from_euler(
                    'xyz', [0, -np.pi/2, 0]
                ).as_quat()
            return self.planner.move_to_pose(
                target_pos, target_orn,
                velocity_scale=velocity_scale
            )
        else:
            # Fix #5: pass target_orn to PyBullet backend.
            # Original code dropped target_orn here, causing PyBullet to always
            # use default orientation regardless of what was requested.
            return self.planner.move_to(target_pos, target_orn)

    def move_home(self):
        """
        Move arm to home position.

        Fix #6/#15: Uses arm.home_position joint angles from config YAML,
        NOT a hardcoded Cartesian position. Original code created a joint-space
        home array but then ignored it and moved to [0, 0.3, 0.5] in Cartesian —
        a genuine bug.

        Config (configs/config.yaml):
            arm:
              home_position: [1.57, 1.57, 1.57, 1.57, 1.57]
        """
        if isinstance(self.planner, MoveIt2Planner):
            return self.planner.move_home()
        else:
            # Load home joint angles from config
            try:
                with open(self._config_path) as f:
                    cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                cfg = {}

            n_revolute = len(self.planner.joint_indices)
            home_joints = cfg.get('arm', {}).get(
                'home_position', [0.0] * n_revolute
            )
            home_joints = np.array(home_joints[:n_revolute], dtype=float)

            # Plan directly in joint space — no IK needed for a joint-space target
            waypoints = self.planner.plan_to_joint_angles(home_joints)
            if not waypoints:
                print("move_home: failed to generate waypoints")
                return False

            self.planner.execute_trajectory(waypoints)
            return True

    def add_table_collision(self, table_height=0.0):
        """
        Add table surface as collision object so planner avoids it.

        For MoveIt2: adds to planning scene (collision-aware).
        For PyBullet: adds physical object (visualisation/contact only —
                      planner does NOT route around it).

        Args:
            table_height: Z height of table surface in base frame (meters)
        """
        if isinstance(self.planner, MoveIt2Planner):
            self.planner.add_collision_box(
                name='table',
                position=[0, 0, table_height - 0.025],
                dimensions=[2.0, 2.0, 0.05]
            )
        else:
            # PyBullet: adds physical object but planner ignores it
            self.planner.add_collision_object(
                position=[0, 0, table_height - 0.025],
                size=(2.0, 2.0, 0.05),
                shape='box'
            )

    def shutdown(self):
        if isinstance(self.planner, PyBulletPlanner):
            self.planner.disconnect()