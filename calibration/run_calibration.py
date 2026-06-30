import cv2
import numpy as np
import glob
from pathlib import Path
import yaml

def run_calibration(image_dir='data/calibration/images',
                    output_file='configs/camera_calibration.yaml'):
    """
    Computes camera intrinsics using a 9x6 ChArUco target board.
    Strictly tailored for OpenCV 4.8 API compliance, data checks, and guards.
    """
    # 9x6 Grid configuration parameters
    squares_x, squares_y = 9, 6
    square_len = 0.025  # 25 mm
    marker_len = 0.0125 # 12.5 mm
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    
    all_corners = []
    all_ids = []
    image_size = None
    
    images = sorted(glob.glob(f"{image_dir}/*.png"))
    if not images:
        raise FileNotFoundError(f"Directory empty or missing: '{image_dir}'. Ensure your dataset is captured first.")
        
    print(f"Processing {len(images)} calibration images using OpenCV {cv2.__version__}...")
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. RESOLUTION CONSISTENCY CHECK Guardrail
        current_size = gray.shape[::-1]  # Formats out to (width, height)
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            print(f"WARNING: Size mismatch in {fname} (Expected {image_size}, got {current_size}) — skipping.")
            continue
            
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        
        # Guardrail check for sufficient geometric structural information
        if charuco_corners is not None and len(charuco_corners) > 15:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        else:
            print(f"Skipping {fname} — insufficient corners detected.")
            
    print(f"\nUsing {len(all_corners)} valid perspectives for calibration.")
    if len(all_corners) < 10:
        raise RuntimeError("Mathematical dataset unsafe. Collect at least 10 high-quality frames.")
        
    # Execute analytical solver via standard OpenCV 4.8 API method
    # Slicing [:5] isolates: ret, camera_matrix, dist_coeffs, rvecs, tvecs
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )[:5]
        
    print(f"\n=== CALIBRATION RESULTS ===")
    print(f"RMS reprojection error: {ret:.4f} pixels")
    print(f"Status Evaluation: {'[EXCELLENT]' if ret < 0.5 else '[ACCEPTABLE]' if ret <= 1.0 else '[REDO DATA COLLECTION]'}")
    
    # Extract variables for physical sanity checks
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    w, h = image_size
    k1 = dist_coeffs.flatten()[0]
    
    # 2. INTRA-MATRIX PHYSICAL SANITY CHECKS
    print("\n--- Geometric Sanity Report ---")
    print(f"  Focal Lengths: fx = {fx:.2f} px | fy = {fy:.2f} px")
    print(f"  fx/fy Aspect Ratio: {(fx/fy):.4f} (Should be close to 1.0)")
    print(f"  Principal Point Offset from Center: ({abs(cx - w/2):.1f}, {abs(cy - h/2):.1f}) px")
    print(f"  Radial Distortion Component (k1): {k1:.4f}")
    
    # Raise flags for suspicious aspect ratios or off-center optical axes
    if abs((fx / fy) - 1.0) > 0.05:
        print("  [WARNING] fx/fy ratio is suspicious! Double check target board metric parameters.")
        
    if abs(cx - w / 2) > 100 or abs(cy - h / 2) > 100:
        print("  [WARNING] Principal point is far from center! Inspect board boundary coverage.")
        
    # 3. DISTORTION BOUNDS CHECK
    if abs(k1) > 0.5:
        print(f"  [WARNING] k1={k1:.4f} is unusually large! Recapture dataset with better coverage at frame edges.")
    print("-------------------------------\n")
    
    # Package into clean dictionary tree format
    calib_data = {
        "rms_error": float(ret),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": camera_matrix.flatten().tolist()
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": dist_coeffs.flatten().tolist()
        }
    }
    
    # Verify and create filesystem paths safely
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(calib_data, f, default_flow_style=False)
        
    print(f"[SUCCESS] Intrinsic metrics written to: {output_file}")
    return camera_matrix, dist_coeffs

def validate_calibration(calib_file='configs/camera_calibration.yaml', device_index=2):
    """
    Live video evaluation showing raw vs rectified image streams side-by-side.
    """
    if not Path(calib_file).exists():
        print(f"[ERROR] Config '{calib_file}' not found. Skipping live validation window.")
        return

    with open(calib_file, 'r') as f:
        calib_data = yaml.safe_load(f)
        
    K = np.array(calib_data['camera_matrix']['data']).reshape((3, 3))
    D = np.array(calib_data['distortion_coefficients']['data']).reshape((1, 5))
    w = calib_data['image_width']
    h = calib_data['image_height']
    
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {device_index}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    # Precompute optimal rectifying mapping coordinates to eliminate loop execution lag
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
    
    print("\n--- VISUAL VERIFICATION STREAM RUNNING ---")
    print("Press [ESC] inside image display frame to shut down workspace loops.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Linear map warping allocation (optimized memory access)
            undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            display = np.hstack([frame, undistorted])
            
            # User Interface typography overlays
            cv2.putText(display, "RAW FEED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, "RECTIFIED PLANE", (frame.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Responsive UI window sizing
            scale = 1280 / display.shape[1]
            display = cv2.resize(display, None, fx=scale, fy=scale)
            
            cv2.imshow('Calibration Validation Panel', display)
            if cv2.waitKey(1) == 27:  # Escape key
                break

    finally:
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_calibration()
    validate_calibration()