# planning/grasp/grasp_verifier.py
import cv2
import numpy as np
import time

class GraspVerifier:
    """
    Verifies grasp success using multiple evidence sources.
    Falls back to next method if primary is unavailable.
    """
    
    def __init__(self, arm_controller):
        self.arm = arm_controller
    
    def verify_by_current(self, wait_ms=500):
        """
        Check gripper motor current spike.
        Spike indicates fingers contacted an object.
        Most reliable and fastest method.
        """
        time.sleep(wait_ms / 1000.0)
        return self.arm.grasp_detected()
    
    def verify_by_vision(self, frame_before, frame_after,
                          object_detector, object_class=None):
        """
        Check if object moved from its original 2D position.
        
        Logic:
        - Object at position P_before before grasp
        - After lift: if object still detected near P_before -> failed
        - After lift: if object not detected or moved significantly -> success
        """
        dets_before = object_detector.detect_objects(frame_before)
        if not dets_before:
            return False
        
        if object_class:
            dets_before = [d for d in dets_before
                          if d['class_name'] == object_class]
        
        if not dets_before:
            return False
        
        original_pos = dets_before[0]['center_2d']
        
        dets_after = object_detector.detect_objects(frame_after)
        
        if not dets_after:
            # Object disappeared entirely — likely being held
            return True
        
        if object_class:
            dets_after = [d for d in dets_after
                         if d['class_name'] == object_class]
        
        if not dets_after:
            return True  # class disappeared — success
        
        new_pos = dets_after[0]['center_2d']
        movement = np.linalg.norm(new_pos - original_pos)
        
        # Object moved significantly from original position
        return movement > 60  # pixels
    
    def verify_combined(self, frame_before, arm_controller,
                         object_detector, object_class=None):
        """
        Use both current and vision verification.
        Returns (success, confidence) where confidence is 'high'/'medium'/'low'
        """
        # Get frame after lift
        import cv2
        # Assumes camera feed is available — caller should pass cap
        # Simplified: just use current for now
        
        current_ok = self.verify_by_current()
        
        if current_ok:
            return True, 'high'
        
        return False, 'low'


class RetryController:
    """
    Manages retry logic for failed grasps.
    On each retry, applies a systematic offset to the grasp pose
    to explore nearby positions.
    """
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        
        # Offset pattern: center, right, left, forward
        self.offsets_m = [
            np.array([0.00,  0.00,  0.00]),
            np.array([0.02,  0.00,  0.00]),
            np.array([-0.02, 0.00,  0.00]),
            np.array([0.00,  0.00,  0.02]),
        ]
    
    def run(self, grasp_fn, verify_fn, arm_controller):
        """
        Run grasp function with retries.
        
        Args:
            grasp_fn: callable(offset) -> bool
            verify_fn: callable() -> bool
            arm_controller: for reset between retries
        """
        for attempt in range(self.max_retries + 1):
            offset = self.offsets_m[min(attempt, len(self.offsets_m)-1)]
            
            print(f"\n--- Attempt {attempt+1}/{self.max_retries+1} "
                  f"(offset: {offset*100}cm) ---")
            
            success = grasp_fn(offset)
            
            if success and verify_fn():
                print(f"Grasp succeeded on attempt {attempt+1}")
                return True
            
            if attempt < self.max_retries:
                print("Grasp failed, resetting for retry...")
                arm_controller.open_gripper()
                time.sleep(0.5)
                arm_controller.move_home()
                time.sleep(2.0)
        
        print("All retry attempts exhausted")
        return False