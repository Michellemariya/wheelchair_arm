from perception.detection.object_detector import ObjectDetector
import cv2

# Load detector
det = ObjectDetector('configs/config.yaml')

# Open webcam
cap = cv2.VideoCapture(2)

print("YOLO + FastSAM Test")
print("Press ESC to quit\n")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Stage 1: YOLO detection
    detections = det.detect_objects(frame)

    if detections:

        # Stage 2: FastSAM segmentation
        masks = det.segment_all_objects(frame, detections)

        # Print mask properties
        print("\nDetected Objects:")

        for d, m in zip(detections, masks):

            if m is not None:

                props = det.get_mask_properties(m)

                print(
                    f"  {d['class_name']:<12}"
                    f" area={props['area_px']:<8}"
                    f" angle={props['angle_rad']:.2f} rad"
                )

        # Draw boxes + masks
        out = det.draw_detections(
            frame,
            detections,
            masks
        )

        cv2.imshow("YOLO + FastSAM", out)

    else:

        cv2.imshow("YOLO + FastSAM", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
