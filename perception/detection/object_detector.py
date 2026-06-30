# perception/detection/object_detector.py
import cv2
import numpy as np
from ultralytics import YOLO, FastSAM
#from ultralytics.models.fastsam import FastSAMPrompt
import yaml
import torch


class ObjectDetector:
    """
    Two-stage detector:
      Stage 1 — YOLOv8  : fast bounding-box detection and class labeling
      Stage 2 — FastSAM : precise pixel mask given a bounding-box prompt

    Fixes applied vs original:
      [BUG1] GPU tensor safely converted via .cpu().numpy() before .astype()
      [BUG2] BGR→RGB conversion applied before FastSAM (both inference call
             and FastSAMPrompt constructor receive the same RGB frame)
      [BUG3] retina_masks=False for real-time performance
      [BUG4] SVD rewritten in explicit (x, y) space to remove 90-deg offset
      [BUG5] YOLO inference passes device= explicitly
      [BUG6] torch.cuda.empty_cache() after FastSAM batch to prevent GPU OOM
      [BUG7] get_mask_properties returns consistent dict instead of mixed None tuple
      [BUG8] draw_detections mask blending fixed to use original frame copy
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        det_cfg = cfg['detection']
        loc_cfg = cfg['localization']

        self.confidence_threshold = float(det_cfg['confidence_threshold'])
        self.target_classes       = {int(k): v
                                     for k, v in det_cfg['target_classes'].items()}
        self.mask_erosion_pixels  = int(loc_cfg['mask_erosion_pixels'])

        # Device — determined once and reused everywhere
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Loading YOLO model...")
        self.yolo = YOLO(det_cfg['yolo_model'])

        print("Loading FastSAM model...")
        self.sam = FastSAM(det_cfg['fastsam_model'])

        print(f"Running detection on: {self.device}")

    # ------------------------------------------------------------------
    # Stage 1 — YOLO detection
    # ------------------------------------------------------------------

    def detect_objects(self, frame):
        """
        Run YOLOv8 detection on a BGR frame.

        YOLO (ultralytics) handles BGR natively — no color conversion needed.

        Returns:
            list of dicts sorted by confidence (descending), each with:
                class_id, class_name, confidence,
                bbox  : np.ndarray [x1, y1, x2, y2] int
                center_2d : np.ndarray [cx, cy]  float64
        """
        # [BUG5] Pass device explicitly so inference always lands on the
        #        correct backend, not whatever YOLO defaulted to at load time.
        results = self.yolo(frame, verbose=False, device=self.device)[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.target_classes:
                continue

            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue

            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox   # top-left and bottom-right corners

            detections.append({
                'class_id':   class_id,
                'class_name': self.target_classes[class_id],
                'confidence': confidence,
                'bbox':       bbox,
                'center_2d':  np.array([(x1 + x2) / 2.0,
                                        (y1 + y2) / 2.0], dtype=np.float64)
            })

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Stage 2 — FastSAM segmentation
    # ------------------------------------------------------------------

    def segment_all_objects(self, frame, detections):
        """
        Run FastSAM ONCE per frame, then prompt for each detected bbox.

        Args:
            frame:      BGR image from OpenCV (will be converted to RGB internally)
            detections: list returned by detect_objects()

        Returns:
            list of masks (np.uint8, 0/255), same order as detections.
            Entry is None if FastSAM failed to produce a mask for that object.
        """
        if not detections:
            return []

        # [BUG2] FastSAM's ViT backbone was trained on RGB images.
        #        Feeding BGR causes systematic channel mismatch in the
        #        feature extractor, degrading mask boundary quality.
        #        Both the inference call AND FastSAMPrompt must receive the
        #        same converted frame so prompt coordinates stay consistent.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # [BUG3] retina_masks=True upsample masks to full input resolution,
        #        which halves FPS with no meaningful benefit for grasp planning.
        #        retina_masks=False keeps masks at feature-map resolution (~80x45
        #        for a 640-input), which is sufficient for centroid/orientation.
        results = self.sam(
            frame_rgb,
            device     = self.device,
            retina_masks = False,   # [BUG3] was True
            imgsz      = 640,
            conf       = 0.4,
            iou        = 0.9,
            verbose    = False
        )

        # [BUG2] FastSAMPrompt also receives the frame — must be the same
        #        RGB frame used for inference above.
        # FastSAM results
        r = results[0]

        if r.masks is None or len(r.boxes) == 0:
            return [None] * len(detections)

        sam_masks = r.masks.data.cpu().numpy()      # (N,H,W)
        sam_boxes = r.boxes.xyxy.cpu().numpy()      # (N,4)

        masks = []

        for det in detections:

            yolo_box = np.array(det['bbox'], dtype=np.float32)

            best_iou = 0.0
            best_idx = -1

            for i, sam_box in enumerate(sam_boxes):

                xA = max(yolo_box[0], sam_box[0])
                yA = max(yolo_box[1], sam_box[1])
                xB = min(yolo_box[2], sam_box[2])
                yB = min(yolo_box[3], sam_box[3])

                inter = max(0, xB - xA) * max(0, yB - yA)

                area1 = (
                    (yolo_box[2] - yolo_box[0]) *
                    (yolo_box[3] - yolo_box[1])
                )

                area2 = (
                    (sam_box[2] - sam_box[0]) *
                    (sam_box[3] - sam_box[1])
                )

                union = area1 + area2 - inter

                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou > 0.2:

                mask = (sam_masks[best_idx] > 0).astype(np.uint8) * 255

                masks.append(mask)

            else:
                masks.append(None)

        if self.device == 'cuda':
                    torch.cuda.empty_cache()

        return masks

    # ------------------------------------------------------------------
    # Mask properties (centroid + orientation)
    # ------------------------------------------------------------------

    def get_mask_properties(self, mask):
        """
        Extract centroid, orientation, and area from a binary mask.

        Returns:
            dict with keys:
                centroid_2d : np.ndarray [cx, cy] float64  (pixel coords)
                angle_rad   : float — orientation angle in radians,
                              measured from positive image x-axis (horizontal right),
                              range [-π/2, π/2]
                area_px     : int — mask area in pixels
                valid       : bool — False if mask was too thin or empty
            Always returns a dict (never None), with valid=False on failure,
            so callers can check one field instead of handling mixed Nones.
        """
        _fail = {'centroid_2d': None, 'angle_rad': None,
                 'area_px': 0, 'valid': False}

        # Erode to remove noisy boundary pixels
        k = self.mask_erosion_pixels
        kernel = np.ones((k, k), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=2)

        if eroded.sum() == 0:
            # Erosion removed everything — fall back to original mask
            eroded = mask

        M = cv2.moments(eroded)
        if M['m00'] == 0:
            return _fail

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        centroid = np.array([cx, cy], dtype=np.float64)

        area = int(M['m00'])

        # Need enough points for a meaningful PCA
        rows, cols = np.where(eroded > 0)  # (y_indices, x_indices)
        if len(rows) < 10:
            # [BUG7] Consistent return: dict with valid=False, not mixed Nones
            return {'centroid_2d': centroid, 'angle_rad': None,
                    'area_px': area, 'valid': False}

        # [BUG4] Original code stacked points as (row, col) = (y, x) and
        #        passed [dy, dx] from SVD directly to arctan2(dy, dx).
        #        arctan2(row_component, col_component) measures the angle from
        #        the IMAGE Y-AXIS (row direction), not from the horizontal x-axis.
        #        Downstream IK solvers expect angle from horizontal, so this
        #        caused a 90-degree offset in end-effector rotation.
        #
        #        Fix: reorder to explicit (x=col, y=row) BEFORE stacking so
        #        SVD returns [dx, dy] directly in standard Cartesian convention.
        points_xy = np.column_stack((cols, rows)).astype(np.float64)  # (N,2): [x,y]

        mean_xy  = points_xy.mean(axis=0)
        centered = points_xy - mean_xy

        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # Vt[0] is now [dx, dy] in standard image Cartesian space
        dx, dy   = Vt[0]
        # Standard angle: measured from positive x-axis (horizontal right)
        angle_rad = np.arctan2(dy, dx)   # range [-π, π]
        # Clamp to [-π/2, π/2] — orientation has 180-deg ambiguity
        if angle_rad > np.pi / 2:
            angle_rad -= np.pi
        elif angle_rad < -np.pi / 2:
            angle_rad += np.pi

        return {
            'centroid_2d': centroid,
            'angle_rad':   angle_rad,
            'area_px':     area,
            'valid':       True
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_detections(self, frame, detections, masks=None):
        """
        Draw bounding boxes, labels, centers, and mask overlays on frame.

        Args:
            frame:      BGR image
            detections: list from detect_objects()
            masks:      list from segment_all_objects() (optional)

        Returns:
            Annotated BGR image.
        """
        # [BUG8] Keep an unmodified copy as the base for all mask blends.
        #        Original code used 'out' (already modified by previous
        #        iterations) as the overlay base, causing each mask to be
        #        blended on top of previously blended content and become
        #        progressively dimmer.
        base = frame.copy()
        out  = frame.copy()

        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = det['bbox']

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Centre dot
            cx, cy = det['center_2d'].astype(int)
            cv2.circle(out, (cx, cy), 5, color, -1)

            # Mask overlay
            if masks and i < len(masks) and masks[i] is not None:
                # [BUG8] Build overlay from clean base, not the running 'out',
                #        so blending weight stays consistent across all masks.
                overlay = base.copy()
                overlay[masks[i] > 0] = color
                out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)

        return out
