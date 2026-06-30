# perception/localization/table_plane.py
import cv2
import numpy as np
import yaml

class TablePlaneLocalizer:
    """
    Localizes objects in 3D by intersecting camera rays with a known table plane.
    
    Assumes objects rest on a flat surface (table, tray, etc.).
    Works entirely from monocular camera + intrinsics — no depth sensor.
    
    Calibration:
        Touch the table with the gripper at 3+ positions.
        Record gripper tip position in camera frame at each touch.
        Fit a plane to those 3D points.
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        calib = np.load(cfg['camera']['calibration_file'])
        self.K = calib['camera_matrix']
        self.K_inv = np.linalg.inv(self.K)
        
        self.plane = None  # [a, b, c, d] for ax+by+cz+d=0
        
        plane_file = cfg['localization']['table_plane_file']
        try:
            self.plane = np.load(plane_file)
            print(f"Table plane loaded: {self.plane}")
        except FileNotFoundError:
            print("Table plane not calibrated. Run calibrate_plane() first.")
    
    def calibrate_plane(self, gripper_tracker, device_index=0,
                        output_file='configs/table_plane.npy'):
        """
        Interactive table plane calibration.
        Touch table with gripper at 4+ positions, press SPACE each time.
        """
        cap = cv2.VideoCapture(device_index)
        calib = np.load('configs/camera_calibration.npz')
        K = calib['camera_matrix']
        D = calib['dist_coeffs']
        
        touch_points = []
        
        print("=== TABLE PLANE CALIBRATION ===")
        print("Touch the table surface with the gripper tip.")
        print("Press SPACE at each touch point. Need 4+ points.")
        print("Place touches at different positions across the workspace.")
        print("ESC when done.\n")
        
        while True:
            ret, frame = cap.read()
            undistorted = cv2.undistort(frame, K, D)
            detection = gripper_tracker.detect(undistorted)
            
            display = gripper_tracker.draw_overlay(undistorted, detection)
            cv2.putText(display,
                       f"Touch points: {len(touch_points)}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            cv2.imshow('Table Calibration', display)
            key = cv2.waitKey(1)
            
            if key == 32 and detection:
                # Gripper tip is touching the table — record its position
                tip = detection['tip_position']
                touch_points.append(tip.copy())
                print(f"Point {len(touch_points)} recorded: "
                      f"{tip*100} cm")
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(touch_points) < 3:
            raise RuntimeError("Need at least 3 touch points")
        
        # Least-squares plane fit to all touch points
        points = np.array(touch_points)
        
        # Fit plane ax+by+cz+d=0 using SVD
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, Vt = np.linalg.svd(centered)
        
        # Normal is last row of Vt (smallest singular value direction)
        normal = Vt[-1]
        normal = normal / np.linalg.norm(normal)
        
        # Ensure normal points toward camera (positive z component)
        if normal[2] < 0:
            normal = -normal
        
        d = -np.dot(normal, centroid)
        self.plane = np.array([normal[0], normal[1], normal[2], d])
        
        np.save(output_file, self.plane)
        print(f"\nTable plane: {self.plane}")
        print(f"Saved to {output_file}")
        
        # Report fit residuals
        residuals = np.abs(points @ normal + d)
        print(f"Fit residuals: mean={residuals.mean()*1000:.1f}mm "
              f"max={residuals.max()*1000:.1f}mm")
        
        return self.plane
    
    def pixel_to_3d(self, u, v):
        """
        Back-project 2D pixel to 3D point on table plane.
        
        Args:
            u, v: pixel coordinates (floats)
        
        Returns:
            numpy array [X, Y, Z] in camera frame (meters), or None
        """
        if self.plane is None:
            return None
        
        # Camera origin in camera frame = [0,0,0]
        cam_origin = np.zeros(3)
        
        # Ray direction through pixel
        pixel_h = np.array([u, v, 1.0])
        ray_dir = self.K_inv @ pixel_h
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Intersect ray with plane
        a, b, c, d = self.plane
        normal = self.plane[:3]
        
        denom = np.dot(normal, ray_dir)
        if abs(denom) < 1e-6:
            return None  # ray is parallel to plane
        
        t = -(np.dot(normal, cam_origin) + d) / denom
        
        if t <= 0:
            return None  # intersection is behind camera
        
        point_3d = cam_origin + t * ray_dir
        return point_3d
    
    def is_calibrated(self):
        return self.plane is not None