# ros2_nodes/joint_bridge_node.py
#
# Thin ROS2 bridge between MoveIt2/visual_servo and ArmController.
#
# ArmController already owns the serial connection, packet protocol,
# feedback parsing, thread safety, and reconnection logic — this node
# does NOT duplicate any of that. It only translates ROS2 topics into
# ArmController method calls and publishes ArmController's state back
# out as a JointState message.
#
# Architecture:
#   MoveIt2 / visual_servo.py
#         ↓ publishes
#   /joint_commands, /cartesian_velocity, /gripper_command
#         ↓ subscribed here
#   joint_bridge_node.py
#         ↓ calls methods on
#   ArmController   (owns serial + ESP32 connection)
#         ↓
#   ESP32
#
# IMPORTANT: ArmController must be instantiated ONLY here. Nothing else
# in the codebase (e.g. visual_servo.py) should create its own
# ArmController() instance — that would open a second serial connection
# to the same port and conflict with this one.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from control.arm.arm_controller import ArmController


class JointBridgeNode(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_flex_joint',
        'elbow_roll_joint',
        'wrist_pitch_joint',
        'wrist_yaw_joint',
        'gripper_joint',
    ]

    N_ARM_JOINTS = 6   # excludes gripper

    def __init__(self):
        super().__init__('joint_bridge_node')

        # fix #6: wrap ArmController construction — config or serial errors
        # would otherwise crash node startup with an unclear traceback.
        try:
            self.arm = ArmController()
        except Exception as e:
            self.get_logger().error(
                f"Failed to initialize ArmController: {e}. "
                "Check configs/config.yaml and serial port settings."
            )
            raise

        # fix #5: track commanded gripper state explicitly. grasp_detected()
        # reports CONTACT (current draw), not open/closed position — using
        # it as gripper position would show a closed-but-empty gripper as
        # "open" in /joint_states, which is wrong.
        self._gripper_closed = False

        self.pub_joint_states = self.create_publisher(
            JointState, '/joint_states', 10
        )

        self.sub_joint_commands = self.create_subscription(
            JointState, '/joint_commands', self.joint_command_callback, 10
        )

        self.sub_velocity = self.create_subscription(
            Twist, '/cartesian_velocity', self.velocity_callback, 10
        )

        self.sub_gripper = self.create_subscription(
            Bool, '/gripper_command', self.gripper_callback, 10
        )

        self.timer = self.create_timer(0.02, self.publish_joint_states)

        self.get_logger().info("Joint bridge node started")

    # ── Command callbacks ────────────────────────────────────────────────────

    def joint_command_callback(self, msg: JointState):
        """
        Forwards arm joint angles to ArmController.

        fix #2: if msg contains a 7th value for gripper_joint (which is how
        MoveIt2 typically publishes JointState — all joints together), route
        it to the gripper instead of silently discarding it.
        """
        if len(msg.position) < self.N_ARM_JOINTS:
            self.get_logger().warn(
                f"Received JointState with {len(msg.position)} positions, "
                f"expected at least {self.N_ARM_JOINTS}"
            )
            return

        try:
            self.arm.send_joint_angles(msg.position[:self.N_ARM_JOINTS])
        except Exception as e:
            self.get_logger().error(f"Failed to send joint command: {e}")
            return

        # If gripper position was included in the same message, honor it too.
        if len(msg.position) >= self.N_ARM_JOINTS + 1:
            gripper_pos = msg.position[self.N_ARM_JOINTS]
            should_close = gripper_pos > 0.5
            self._command_gripper(should_close)

    def velocity_callback(self, msg: Twist):
        velocity_xyz = [msg.linear.x, msg.linear.y, msg.linear.z]
        try:
            self.arm.send_velocity(velocity_xyz)
        except Exception as e:
            self.get_logger().error(f"Velocity command failed: {e}")

    def gripper_callback(self, msg: Bool):
        self._command_gripper(msg.data)

    def _command_gripper(self, close: bool):
        """Single place that issues gripper commands and tracks state (fix #5)."""
        try:
            if close:
                self.arm.close_gripper()
            else:
                self.arm.open_gripper()
            self._gripper_closed = close
        except Exception as e:
            self.get_logger().error(f"Gripper command failed: {e}")

    # ── State publishing ─────────────────────────────────────────────────────

    def publish_joint_states(self):
        """
        Publishes current arm + gripper state to /joint_states.

        fix #4: validates joint_positions length before building the message,
        since a name/position length mismatch produces a malformed
        JointState that can cause undefined behavior in MoveIt2/RViz.

        fix #5: gripper position comes from tracked commanded state
        (_gripper_closed), NOT grasp_detected() — the latter measures
        contact/current draw, not open/closed position.
        """
        try:
            joint_positions = self.arm.get_joint_states()
        except Exception as e:
            self.get_logger().error(f"Failed to read joint states: {e}")
            return

        if len(joint_positions) != self.N_ARM_JOINTS:
            self.get_logger().error(
                f"get_joint_states() returned {len(joint_positions)} values, "
                f"expected {self.N_ARM_JOINTS}. Skipping publish to avoid "
                f"malformed JointState message."
            )
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.JOINT_NAMES

        gripper_position = 1.0 if self._gripper_closed else 0.0
        msg.position = list(joint_positions) + [gripper_position]

        try:
            self.pub_joint_states.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Joint state publish failed: {e}")

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def destroy_node(self):
        try:
            self.arm.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JointBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# fix #1: was "if name == 'main':" — undefined variables, would crash on run
if __name__ == '__main__':
    main()