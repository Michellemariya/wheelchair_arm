from perception.detection.object_detector import ObjectDetector
import cv2

det = ObjectDetector('configs/config.yaml')

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = det.detect_objects(frame)

    print(f"Detections: {len(detections)}", end="\r")

    out = det.draw_detections(frame, detections, None)

    cv2.imshow("YOLO", out)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

