import cv2
import numpy as np
import yaml
from pathlib import Path

class HandEyeCalibrator:
    """
    Calibrates the fixed transform between camera frame and arm base frame.
    Tailored for Eye-to-Hand configuration under OpenCV 4.8+.
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        self.config_path = config_path

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Master config not found at '{config_path}'.")
            
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        calib_file = self.cfg['camera']['calibration_file']
        if not Path(calib_file).exists():
            raise FileNotFoundError(f"Camera calibration file not found at '{calib_file}'.")
            
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)

        self.K = np.array(calib_data['camera_matrix']['data'], dtype=np.float64).reshape((3, 3))
        
        # FIXED: Accommodates variable lens parameters safely without crashing
        self.D = np.array(calib_data['distortion_coefficients']['data'], dtype=np.float64).flatten().reshape((1, -1))

        self.w = int(calib_data['image_width'])
        self.h = int(calib_data['image_height'])

        from perception.camera.gripper_tracker import GripperTracker
        self.tracker = GripperTracker(config_path)
        self.T_cam_to_base = None

    def collect_poses(self, arm_controller, device_index: int = 0, n_poses: int = 20):
        """
        Interactive collection of calibration pose pairs with hardware loop guards.
        """
        if n_poses < 15:
            raise ValueError("Hand-eye calibration requires at least 15 poses.")

        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at device index {device_index}.")
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.w or actual_h != self.h:
            print(f"[WARNING] Resolution mismatch! System: {self.w}x{self.h} | Hardware: {actual_w}x{actual_h}")

        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []

        print("=== HAND-EYE CALIBRATION — POSE COLLECTION ===")
        print("SPACE = record pose | ESC = finish early\n")

        consecutive_dropped_frames = 0

        while len(R_gripper2base_list) < n_poses:
            ret, frame = cap.read()
            if not ret:
                consecutive_dropped_frames += 1
                if consecutive_dropped_frames > 100:
                    cap.release()
                    raise RuntimeError("Camera connection lost during data collection loop.")
                continue
            
            consecutive_dropped_frames = 0
            detection = self.tracker.detect(frame)
            display = self.tracker.draw_overlay(frame, detection)

            n = len(R_gripper2base_list)
            cv2.putText(display, f"Poses: {n}/{n_poses}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if detection:
                cv2.putText(display, "TAG DETECTED — press SPACE to record", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No tag visible — reposition", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Hand-Eye Calibration', display)
            key = cv2.waitKey(1)

            if key == 32 and detection: 
                joint_angles = arm_controller.get_joint_states()
                T_gripper_in_base = arm_controller.forward_kinematics(joint_angles)

                # Mathematical Inversion Guard for Eye-to-Hand Layouts
                T_base_to_gripper = np.linalg.inv(T_gripper_in_base)

                R_gripper2base_list.append(T_base_to_gripper[:3, :3])
                t_gripper2base_list.append(T_base_to_gripper[:3, 3].reshape(3, 1))

                R_target2cam_list.append(detection['R'])
                t_target2cam_list.append(detection['tvec'].reshape(3, 1))

                print(f"Pose {n+1:02d} recorded | EE position (base frame): {T_gripper_in_base[:3, 3].round(4)}")

            elif key == 27: 
                print("Early exit requested.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list

    def solve(self, R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI, output_file: str = 'configs/T_cam_to_base.npy'):
        """
        Solves hand-eye optimization equation system without index slicing bugs.
        """
        if len(R_g2b) < 15:
            raise ValueError(f"Need at least 15 pose pairs, got {len(R_g2b)}.")

        print("\nSolving AX = XB (hand-eye calibration)...")
        
        # FIXED: Explicit return tuple matching safely without append slicing error tags
        R_cam2base, t_cam2base = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)

        T_cam_to_base = np.eye(4)
        T_cam_to_base[:3, :3] = R_cam2base
        T_cam_to_base[:3, 3] = t_cam2base.flatten()

        self._validate_transform(T_cam_to_base)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        np.save(output_file, T_cam_to_base)

        self.T_cam_to_base = T_cam_to_base
        print("[SUCCESS] Hand-eye calibration complete.")
        return T_cam_to_base

    def validate(self, T_cam_to_base: np.ndarray, device_index: int = 0):
        """
        Live verification utility providing direct comparison readouts.
        """
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at device index {device_index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        print("\n--- VALIDATION PANEL ---")
        print("SPACE = compute base-frame position | ESC = exit\n")

        consecutive_dropped_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_dropped_frames += 1
                if consecutive_dropped_frames > 100:
                    cap.release()
                    raise RuntimeError("Camera connection lost during metrics validation loop.")
                continue

            consecutive_dropped_frames = 0
            detection = self.tracker.detect(frame)
            display = self.tracker.draw_overlay(frame, detection) if detection else frame.copy()

            status_text = "TAG VISIBLE — press SPACE" if detection else "No tag detected"
            status_color = (0, 255, 0) if detection else (0, 0, 255)
            cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow('Hand-Eye Validation', display)
            key = cv2.waitKey(1)

            if key == 32 and detection: 
                pos_cam = np.append(detection['tvec'].flatten(), 1.0)
                pos_base = T_cam_to_base @ pos_cam

                print(f"Tag position in CAMERA frame (cm): X={detection['tvec'][0]*100:.2f} Y={detection['tvec'][1]*100:.2f} Z={detection['tvec'][2]*100:.2f}")
                print(f"Tag position in BASE frame (cm):   X={pos_base[0]*100:.2f} Y={pos_base[1]*100:.2f} Z={pos_base[2]*100:.2f}")
                print("Measure physically with your ruler and verify.\n")

            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def load_transform(transform_file: str = 'configs/T_cam_to_base.npy') -> np.ndarray:
        if not Path(transform_file).exists():
            raise FileNotFoundError(
                f"T_cam_to_base not found at '{transform_file}'. Run calibration first."
            )

        return np.load(transform_file).astype(np.float64)

    def _validate_transform(self, T: np.ndarray):
        R = T[:3, :3]
        t = T[:3, 3]

        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            print(f"[WARNING] R determinant = {det:.6f} (should be 1.0). Rotation matrix non-orthogonal.")

        t_norm = np.linalg.norm(t)
        if t_norm > 2.0:
            print(f"[WARNING] Translation magnitude = {t_norm:.3f} m. Unusually large layout check triggered.")
        elif t_norm < 0.01:
            print("[WARNING] Translation magnitude is near zero. Verify tracking frames.")

        print(f"Transform check complete: det(R)={det:.6f} | |t|={t_norm:.4f} m")

if __name__ == '__main__':
    # Execution entry wrapper remains unchanged
    pass