from perception.detection.object_detector import ObjectDetector
from perception.keypoints.keypoint_localizer import SemanticKeypointExtractor
import cv2

det = ObjectDetector('configs/config.yaml')
kp = SemanticKeypointExtractor('configs/config.yaml')

cap = cv2.VideoCapture(2)

print("Press SPACE to detect + extract keypoints")
print("Press ESC to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("live", frame)

    key = cv2.waitKey(30)

    if key == 27:
        break

    if key == 32:

        detections = det.detect_objects(frame)

        if not detections:
            print("No objects detected")
            continue

        masks = det.segment_all_objects(frame, detections)

        out = frame.copy()

        for d, m in zip(detections, masks):

            if m is None:
                continue

            kps_2d = kp.detect_rule_based(
                d['class_name'],
                m,
                frame
            )

            print(f"\n{d['class_name']} keypoints:")

            for name, (u, v) in kps_2d.items():

                print(f"  {name}: ({u:.0f}, {v:.0f})")

                cv2.circle(
                    out,
                    (int(u), int(v)),
                    6,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    out,
                    name,
                    (int(u)+8, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )

        cv2.imshow("keypoints", out)

cap.release()
cv2.destroyAllWindows()

