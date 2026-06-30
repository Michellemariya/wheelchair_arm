import numpy as np


class ArmKinematics:
    """
    Forward kinematics for wheelchair arm using
    Modified Denavit-Hartenberg (Craig) convention.

    Coordinate convention:
        Base Frame
            +X forward
            +Y left
            +Z up

    Units:
        metres
        radians

    Modified DH:
        Rot(x, alpha)
            ↓
        Trans(x, a)
            ↓
        Rot(z, theta)
            ↓
        Trans(z, d)
    """

    def __init__(self):

        # =====================================================
        # MODIFIED DH PARAMETERS
        #
        # Units: metres
        # Angles: radians
        #
        # [a, alpha, d, theta_offset]
        # =====================================================

        self.dh_params = [

            # Joint 1
            {
                "a": 0.0,
                "alpha": np.deg2rad(0.0),
                "d": 0.0,
                "theta_offset": 0.0
            },

            # Joint 2
            {
                "a": 29.317e-3,
                "alpha": np.deg2rad(90.0),
                "d": 37.776e-3,
                "theta_offset": 0.0
            },

            # Joint 3
            {
                "a": 148.879e-3,
                "alpha": np.deg2rad(0.0),
                "d": 37.115e-3,
                "theta_offset": 0.0
            },

            # Joint 4
            {
                "a": 62.879e-3,
                "alpha": np.deg2rad(0.0),
                "d": 37.213e-3,
                "theta_offset": 0.0
            },

            # Joint 5
            {
                "a": 84.900e-3,
                "alpha": np.deg2rad(90.0),
                "d": -0.554e-3,
                "theta_offset": 0.0
            },

            # Joint 6
            {
                "a": 49.750e-3,
                "alpha": np.deg2rad(0.0),
                "d": 0.251e-3,
                "theta_offset": 0.0
            }
        ]

        # =====================================================
        # TOOL TRANSFORM
        #
        # Wrist frame -> grasp point
        #
        # Set later after gripper geometry is finalized.
        # =====================================================

        self.T_tool = np.eye(4, dtype=np.float64)

        self.T_tool[:3, 3] = [
            0.0,
            0.0,
            0.0
        ]

    # =========================================================
    # Modified DH Transform
    # =========================================================

    @staticmethod
    def mdh_transform(a, alpha, d, theta):
        """
        Modified DH transform (Craig convention)

        A_i =
            RotX(alpha)
            TransX(a)
            RotZ(theta)
            TransZ(d)
        """

        ct = np.cos(theta)
        st = np.sin(theta)

        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct,      -st,       0.0,      a],
            [st*ca,   ct*ca,    -sa,   -d*sa],
            [st*sa,   ct*sa,     ca,    d*ca],
            [0.0,     0.0,      0.0,    1.0]
        ], dtype=np.float64)

    # =========================================================
    # Forward Kinematics
    # =========================================================

    def forward_kinematics(self, joint_angles):
        """
        Parameters
        ----------
        joint_angles : iterable
            Joint angles in radians.

        Returns
        -------
        T_gripper_in_base : (4,4)
            Homogeneous transform from base frame
            to end-effector frame.
        """

        if len(joint_angles) != len(self.dh_params):
            raise ValueError(
                f"Expected {len(self.dh_params)} joints, "
                f"received {len(joint_angles)}."
            )

        T = np.eye(4, dtype=np.float64)

        for q, p in zip(joint_angles, self.dh_params):

            theta = q + p["theta_offset"]

            A = self.mdh_transform(
                a=p["a"],
                alpha=p["alpha"],
                d=p["d"],
                theta=theta
            )

            T = T @ A

        T = T @ self.T_tool

        return T

    # =========================================================
    # Convenience Functions
    # =========================================================

    def end_effector_position(self, joint_angles):
        """
        Returns:
            np.ndarray (3,)
        """

        T = self.forward_kinematics(joint_angles)

        return T[:3, 3].copy()

    def end_effector_rotation(self, joint_angles):
        """
        Returns:
            np.ndarray (3,3)
        """

        T = self.forward_kinematics(joint_angles)

        return T[:3, :3].copy()

    def end_effector_pose(self, joint_angles):
        """
        Returns:
            position, rotation
        """

        T = self.forward_kinematics(joint_angles)

        position = T[:3, 3].copy()
        rotation = T[:3, :3].copy()

        return position, rotation

    # =========================================================
    # Jacobian Placeholder
    # =========================================================

    def jacobian(self, joint_angles):
        """
        Implement later for:
            - visual servoing
            - differential IK
            - velocity control
        """

        raise NotImplementedError(
            "Jacobian not implemented yet."
        )

    # =========================================================
    # Inverse Kinematics Placeholder
    # =========================================================

    def inverse_kinematics(
        self,
        target_position,
        target_rotation=None
    ):
        """
        Future options:

            - Analytical IK
            - Numerical IK
            - PyBullet IK
            - MoveIt2 IK
        """

        raise NotImplementedError(
            "Inverse kinematics not implemented yet."
        )


if __name__ == "__main__":

    kin = ArmKinematics()

    q = np.deg2rad([
        0,
        0,
        0,
        0,
        0,
        0
    ])

    T = kin.forward_kinematics(q)

    print("Forward Kinematics:")
    print(T)

    print("\nPosition (m):")
    print(T[:3, 3])