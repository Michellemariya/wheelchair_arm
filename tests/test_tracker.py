from perception.camera.gripper_tracker import GripperTracker

def main():
    print("Creating tracker...")
    tracker = GripperTracker()

    print("Starting AprilTag + PnP validation...")
    tracker.validate_accuracy(device_index=2)

if __name__ == "__main__":
    main()
