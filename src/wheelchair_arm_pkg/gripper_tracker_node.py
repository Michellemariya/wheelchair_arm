# ros2_nodes/gripper_tracker_node.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
import tf2_ros

from perception.camera.gripper_tracker import GripperTracker


class GripperTrackerNode(Node):
    """
    ROS2 node: detects the AprilTag mounted on the gripper and publishes:
        1. PoseStamped  → /gripper/pose          (pose in camera optical frame)
        2. TF transform → camera_frame → gripper_tag_link
        3. Float32      → /gripper/tag_margin    (detection quality metric)

    Coordinate convention:
        All poses are published in the camera OPTICAL frame (REP-103):
            +X right, +Y down, +Z forward  (OpenCV convention)
        The frame_id is set to the optical frame ID (e.g. camera_color_optical_frame).
        Downstream nodes (MoveIt2, visual servo) apply T_cam_to_base as needed.
    """

    def __init__(self):
        super().__init__('gripper_tracker_node')

        # ── Parameter declarations ─────────────────────────────────────────
        self.declare_parameter('image_topic',         '/camera/rgb')
        self.declare_parameter('pose_topic',          '/gripper/pose')
        self.declare_parameter('margin_topic',        '/gripper/tag_margin')
        self.declare_parameter('camera_frame_id',     'camera_color_optical_frame')
        self.declare_parameter('gripper_frame_id',    'gripper_tag_link')
        self.declare_parameter('min_decision_margin', 15.0)
        self.declare_parameter('config_path',         'configs/config.yaml')

        image_topic   = self.get_parameter('image_topic').get_parameter_value().string_value
        pose_topic    = self.get_parameter('pose_topic').get_parameter_value().string_value
        margin_topic  = self.get_parameter('margin_topic').get_parameter_value().string_value
        
        # Stored explicitly as instance variables for tracking architecture
        self.config_path   = self.get_parameter('config_path').get_parameter_value().string_value
        self.camera_frame  = self.get_parameter('camera_frame_id').get_parameter_value().string_value
        self.gripper_frame = self.get_parameter('gripper_frame_id').get_parameter_value().string_value
        self.min_margin    = self.get_parameter('min_decision_margin').get_parameter_value().double_value

        # BUG FIX 6: Frame sanity warning gate
        if not self.camera_frame.endswith('_optical_frame'):
            self.get_logger().warn(
                f"camera_frame_id='{self.camera_frame}' does not end with "
                "'_optical_frame'. PnP output is in OpenCV optical convention. "
                "Incorrect frame label will cause RViz/MoveIt2 to misrender poses."
            )

        # ── Core pipeline ──────────────────────────────────────────────────
        # BUG FIX 7: Explicitly configuring configuration path to mirror master config
        self.tracker = GripperTracker(config_path=self.config_path)
        self.bridge  = CvBridge()

        # ── TF broadcaster ─────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── QoS profile ────────────────────────────────────────────────────
        # BUG FIX 8: Real-time sensor stream configuration
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            sensor_qos
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self.pub_pose = self.create_publisher(PoseStamped, pose_topic, 10)
        
        # BUG FIX 9: Publish explicit quality diagnostic margin topic
        self.pub_margin = self.create_publisher(Float32, margin_topic, 10)

        # ── Detection statistics ───────────────────────────────────────────
        # BUG FIX 10: Structural loop rate monitoring variables
        self._total_frames    = 0
        self._detected_frames = 0
        self._dropped_margin  = 0
        self.create_timer(10.0, self._log_detection_stats)

        self.get_logger().info("GripperTrackerNode initialized successfully.")

    def image_callback(self, msg: Image):
        self._total_frames += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge conversion failed: {e}")
            return

        detection = self.tracker.detect(frame)
        if detection is None:
            return

        self._detected_frames += 1

        # ── Quality gate ───────────────────────────────────────────────────
        margin = float(detection.get('decision_margin', 100.0))

        margin_msg       = Float32()
        margin_msg.data  = margin
        self.pub_margin.publish(margin_msg)

        if margin < self.min_margin:
            self._dropped_margin += 1
            return

        # ── Extract position ───────────────────────────────────────────────
        tvec = np.array(detection['tvec'], dtype=np.float64).flatten()

        # ── Extract orientation — SO(3) Orthogonality Checking ─────────────
        R_mat = np.array(detection['R'], dtype=np.float64)

        # BUG FIX 12 HARDENING: Full Orthogonality Validation (R*R^T ≈ I)
        det_R = np.linalg.det(R_mat)
        ortho_check = np.dot(R_mat, R_mat.T)
        identity_error = np.linalg.norm(ortho_check - np.eye(3))
        
        if abs(det_R - 1.0) > 0.05 or identity_error > 0.05:
            self.get_logger().warn(
                f"Degenerate non-SO(3) matrix encountered. Det: {det_R:.4f}, Ortho Error: {identity_error:.4f}. Dropping frame."
            )
            return

        # BUG FIX 11: Flatten safely to secure shape extraction across variations
        quat = Rotation.from_matrix(R_mat).as_quat().flatten()  # Guaranteed [x, y, z, w]
        qx, qy, qz, qw = quat

        # ── Timestamp ─────────────────────────────────────────────────────
        # BUG FIX 13: Preserve incoming image hardware tracking capture stamp
        stamp = msg.header.stamp

        # ── Publish PoseStamped ────────────────────────────────────────────
        pose_msg                    = PoseStamped()
        pose_msg.header.stamp       = stamp
        pose_msg.header.frame_id    = self.camera_frame

        pose_msg.pose.position.x    = float(tvec[0])
        pose_msg.pose.position.y    = float(tvec[1])
        pose_msg.pose.position.z    = float(tvec[2])

        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        self.pub_pose.publish(pose_msg)

        # ── Broadcast TF ───────────────────────────────────────────────────
        t_msg                            = TransformStamped()
        t_msg.header.stamp               = stamp
        t_msg.header.frame_id            = self.camera_frame
        t_msg.child_frame_id             = self.gripper_frame

        t_msg.transform.translation.x    = float(tvec[0])
        t_msg.transform.translation.y    = float(tvec[1])
        t_msg.transform.translation.z    = float(tvec[2])

        t_msg.transform.rotation.x       = float(qx)
        t_msg.transform.rotation.y       = float(qy)
        t_msg.transform.rotation.z       = float(qz)
        t_msg.transform.rotation.w       = float(qw)

        self.tf_broadcaster.sendTransform(t_msg)

    def _log_detection_stats(self):
        if self._total_frames == 0:
            return

        det_rate     = 100.0 * self._detected_frames / self._total_frames
        margin_drop  = self._dropped_margin

        self.get_logger().info(
            f"[Stats] Frames: {self._total_frames} | "
            f"Detected: {self._detected_frames} ({det_rate:.1f}%) | "
            f"Dropped (low margin): {margin_drop}"
        )

        # Reset counters for next window
        self._total_frames    = 0
        self._detected_frames = 0
        self._dropped_margin  = 0


def main(args=None):
    rclpy.init(args=args)
    node = GripperTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()