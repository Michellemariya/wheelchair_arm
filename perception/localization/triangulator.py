# perception/localization/triangulator.py
import cv2
import numpy as np
import yaml

class MotionTriangulator:
    """
    Triangulates 3D object position from two observations at different
    arm configurations. Uses arm motion as a stereo baseline.
    
    More accurate for objects not on the table plane,
    but requires deliberately moving the arm to two positions.
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        calib = np.load(cfg['camera']['calibration_file'])
        self.K = calib['camera_matrix']
        
        # Projection matrix for camera at origin
        self.P_ref = self.K @ np.eye(3, 4)
    
    def triangulate(self,
                    pixel_obs1, pixel_obs2,
                    T_tag_cam1, T_tag_cam2):
        """
        Triangulate object 3D position from two frames.
        
        Args:
            pixel_obs1: [u, v] object pixel in frame 1
            pixel_obs2: [u, v] object pixel in frame 2
            T_tag_cam1: 4x4 tag-in-camera transform at frame 1 (from PnP)
            T_tag_cam2: 4x4 tag-in-camera transform at frame 2
        
        Returns:
            3D position in camera frame at time of frame 1, or None
        """
        # Compute relative camera motion between frames
        # Camera is fixed, but we use gripper pose change as reference
        # For eye-to-hand: camera doesn't move.
        # We need a different approach — compute object-relative to gripper.
        
        # Relative transform from pose 1 to pose 2
        T_rel = np.linalg.inv(T_tag_cam2) @ T_tag_cam1
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3,  3]
        
        # Second projection matrix
        P2 = self.K @ np.hstack([R_rel, t_rel.reshape(3,1)])
        
        # Format pixel observations
        pt1 = np.array([[pixel_obs1[0]], [pixel_obs1[1]]], dtype=np.float64)
        pt2 = np.array([[pixel_obs2[0]], [pixel_obs2[1]]], dtype=np.float64)
        
        # Triangulate
        point_4d = cv2.triangulatePoints(self.P_ref, P2, pt1, pt2)
        
        if abs(point_4d[3]) < 1e-10:
            return None
        
        point_3d = (point_4d[:3] / point_4d[3]).flatten()
        
        # Sanity check: point should be in front of camera
        if point_3d[2] <= 0:
            return None
        
        return point_3d
    
    def collect_two_observations(self, detector, gripper_tracker,
                                  arm_controller, device_index=0,
                                  move_distance=0.05):
        """
        Automatically collect two observations by moving arm slightly.
        """
        cap = cv2.VideoCapture(device_index)
        calib = np.load('configs/camera_calibration.npz')
        K = calib['camera_matrix']
        D = calib['dist_coeffs']
        
        observations = []
        
        # Position 1: current position
        ret, frame = cap.read()
        undistorted = cv2.undistort(frame, K, D)
        
        dets = detector.detect_objects(undistorted)
        tag_det = gripper_tracker.detect(undistorted)
        
        if not dets or tag_det is None:
            cap.release()
            return None, None, None, None
        
        obs1_pixel = dets[0]['center_2d']
        T1 = tag_det['T_tag_cam'].copy()
        
        # Move arm laterally
        arm_controller.send_velocity(
            np.array([move_distance, 0, 0])
        )
        import time; time.sleep(1.0)
        arm_controller.stop()
        time.sleep(0.3)
        
        # Position 2: after move
        ret, frame = cap.read()
        undistorted = cv2.undistort(frame, K, D)
        
        dets2 = detector.detect_objects(undistorted)
        tag_det2 = gripper_tracker.detect(undistorted)
        
        if not dets2 or tag_det2 is None:
            cap.release()
            return None, None, None, None
        
        obs2_pixel = dets2[0]['center_2d']
        T2 = tag_det2['T_tag_cam'].copy()
        
        cap.release()
        return obs1_pixel, obs2_pixel, T1, T2