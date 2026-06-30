from perception.depth.your_dmd_model import MonocularDepthModel
import cv2
import numpy as np

model = MonocularDepthModel()
model.warmup()

cap = cv2.VideoCapture(2)   # your webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    depth = model.infer(frame)

    depth_vis = (depth * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB", frame)
    cv2.imshow("Depth", depth_vis)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
