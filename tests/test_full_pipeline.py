import cv2
import time
import numpy as np

from perception.camera.gripper_tracker import GripperTracker
from perception.depth.scaled_depth import ScaledDepthLocalizer
from perception.depth.your_dmd_model import MonocularDepthModel
from perception.detection.object_detector import ObjectDetector
from perception.keypoints.keypoint_localizer import SemanticKeypointExtractor

CONFIG = "configs/config.yaml"

print("Loading modules...")

tracker = GripperTracker(CONFIG)

depth_model = MonocularDepthModel()
depth_model.warmup()

depth_localizer = ScaledDepthLocalizer(CONFIG)
depth_localizer.set_depth_model(depth_model)

detector = ObjectDetector(CONFIG)
kp_ext = SemanticKeypointExtractor(CONFIG)

print("\nAll modules loaded successfully.")

cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n===================================")
print("FULL PERCEPTION PIPELINE TEST")
print("===================================")
print("Place:")
print("  • AprilTag ID=0")
print("  • Cup or bottle")
print("")
print("SPACE = run one perception cycle")
print("ESC   = quit")
print("")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("live", frame)

    key = cv2.waitKey(30)

    if key == 27:
        break

    if key != 32:
        continue

    print("\n==============================")
    print("NEW PIPELINE CYCLE")
    print("==============================")

    t0 = time.perf_counter()

    # --------------------------------------------------
    # STEP 1 : APRILTAG
    # --------------------------------------------------

    tag = tracker.detect(frame)

    if tag is None:
        print("✗ AprilTag not detected")
        continue

    print(
        f"✓ AprilTag detected | "
        f"Depth = {tag['tvec'][2]*100:.1f} cm"
    )

    # --------------------------------------------------
    # STEP 2 : DEPTH MAP
    # --------------------------------------------------

    depth_map = depth_localizer.get_scaled_depth_map(
        frame,
        tag['tvec'],
        tag['center_2d']
    )

    if depth_map is None:
        print("✗ Depth map generation failed")
        continue

    print(
        f"✓ Depth map generated | "
        f"Range = [{depth_map.min():.2f}, "
        f"{depth_map.max():.2f}] m"
    )

    u_tag = int(tag['center_2d'][0])
    v_tag = int(tag['center_2d'][1])

    print("\n===== DEPTH VALIDATION =====")
    print(f"PnP depth        : {tag['tvec'][2]:.4f} m")
    print(f"Scaled depth map : {depth_map[v_tag, u_tag]:.4f} m")
    print("============================\n")

    # --------------------------------------------------
    # STEP 3 : YOLO
    # --------------------------------------------------

    detections = detector.detect_objects(frame)

    if not detections:
        print("✗ No objects detected")
        continue

    print(
        f"✓ Objects detected: "
        f"{[d['class_name'] for d in detections]}"
    )

    # ==========================================
    # DEBUG YOLO CENTER POINTS
    # ==========================================

    debug = frame.copy()

    for d in detections:
        u, v = d['center_2d']
        u = int(u)
        v = int(v)

        print(
            f"{d['class_name']} center = ({u}, {v})"
        )

        print(
            f"{d['class_name']} scaled depth value = "
            f"{depth_map[v, u]:.6f}"
        )

        cv2.circle(
            debug,
            (u, v),
            10,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            debug,
            d['class_name'],
            (u + 10, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imshow("debug_centers", debug)

    # --------------------------------------------------
    # STEP 4 : FASTSAM
    # --------------------------------------------------

    masks = detector.segment_all_objects(
        frame,
        detections
    )

    print("✓ Segmentation complete")

    out = detector.draw_detections(
        frame,
        detections,
        masks
    )

    # --------------------------------------------------
    # STEP 5 : 3D LOCALIZATION
    # --------------------------------------------------

    for d, mask in zip(detections, masks):

        if mask is None:
            continue

        u, v = d['center_2d']

        u = int(u)
        v = int(v)

        print(f"\n{d['class_name']} center pixel = ({u}, {v})")

        cv2.circle(
            out,
            (u, v),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            out,
            "DEPTH",
            (u + 10, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        # ==========================================
        # MASK DEPTH (MEDIAN)
        # ==========================================

        depth_values = depth_map[mask > 0]

        if len(depth_values) == 0:
            print("No valid depth values in mask")
            continue

        object_depth = float(np.median(depth_values))

        print(
            f"{d['class_name']} mask depth = "
            f"{object_depth*100:.1f} cm"
        )

        print(
            f"Depth stats: "
            f"min={depth_values.min():.3f} "
            f"max={depth_values.max():.3f} "
            f"median={np.median(depth_values):.3f}"
        )

        # ==========================================
        # 3D POSITION
        # ==========================================

        fx = depth_localizer.fx
        fy = depth_localizer.fy
        cx = depth_localizer.cx
        cy = depth_localizer.cy

        X = (u - cx) * object_depth / fx
        Y = (v - cy) * object_depth / fy
        Z = object_depth

        pos_cam = np.array([X, Y, Z])

        print(f"\nObject: {d['class_name']}")

        print(
            f"  Camera XYZ (cm): "
            f"({pos_cam[0]*100:.1f}, "
            f"{pos_cam[1]*100:.1f}, "
            f"{pos_cam[2]*100:.1f})"
        )

        # ----------------------------------------------
        # KEYPOINTS
        # ----------------------------------------------

        kps = kp_ext.detect_rule_based(
            d['class_name'],
            mask,
            frame
        )

        print(
            f"  Keypoints: "
            f"{list(kps.keys())}"
        )

    dt = time.perf_counter() - t0

    print(f"\nCycle time: {dt:.2f}s")

    out = tracker.draw_overlay(
        out,
        tag
    )

    cv2.imshow(
        "pipeline output",
        out
    )

cap.release()
cv2.destroyAllWindows()
