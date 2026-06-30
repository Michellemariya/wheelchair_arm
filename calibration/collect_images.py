# calibration/collect_images.py
import cv2
import numpy as np
import os
import yaml

def collect_calibration_images(config_path='configs/config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    cam_cfg = cfg['camera']
    cap = cv2.VideoCapture(cam_cfg['device_index'])
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((9, 6), 0.025, 0.0125, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    
    save_dir = 'data/calibration/images'
    os.makedirs(save_dir, exist_ok=True)
    
    captured = 0
    target = 30  # collect 30 images minimum
    
    print("=== CALIBRATION IMAGE COLLECTION ===")
    print("Move the board to different positions, angles, and distances.")
    print("Cover all corners of the frame.")
    print("Tilt the board in X and Y directions.")
    print("SPACE = capture | ESC = done")
    print(f"Target: {target} images\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        
        display = frame.copy()
        
        if charuco_corners is not None and len(charuco_corners) > 10:
            cv2.aruco.drawDetectedCornersCharuco(
                display, charuco_corners, charuco_ids, (0, 255, 0)
            )
            # Green border = good detection
            cv2.rectangle(display, (0,0), 
                         (display.shape[1]-1, display.shape[0]-1),
                         (0, 255, 0), 5)
        else:
            # Red border = no detection
            cv2.rectangle(display, (0,0),
                         (display.shape[1]-1, display.shape[0]-1),
                         (0, 0, 255), 5)
        
        cv2.putText(display, f"Captured: {captured}/{target}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Calibration Collection', display)
        
        key = cv2.waitKey(1)
        
        if key == 32:  # SPACE
            if charuco_corners is not None and len(charuco_corners) > 10:
                filename = f"{save_dir}/calib_{captured:03d}.png"
                cv2.imwrite(filename, frame)
                captured += 1
                print(f"Captured image {captured}")
            else:
                print("Board not detected well — reposition and try again")
        
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCollection done. {captured} images saved to {save_dir}/")
    return captured

if __name__ == '__main__':
    collect_calibration_images()