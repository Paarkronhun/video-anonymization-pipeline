import cv2
import numpy as np
import logging
import torch

from ultralytics import YOLO

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# POSE KEYPOINT INDICES (COCO format)
# ==========================================================

KP_NOSE        = 0
KP_LEFT_EYE    = 1
KP_RIGHT_EYE   = 2
KP_LEFT_EAR    = 3
KP_RIGHT_EAR   = 4
KP_LEFT_SHLDR  = 5
KP_RIGHT_SHLDR = 6

# ==========================================================
# COLOR CONSTANTS  (BGR)
# ==========================================================

COLOR_ADULT = (0,   0,   0)    # Black  — adult
COLOR_CHILD = (255, 255, 255)  # White  — child

# ----------------------------------------------------------
# AGE CLASSIFICATION TUNING
#
# Strategy: aspect ratio of the bounding box (width / height).
#
# Adults tend to have broader shoulders → wider bbox relative
# to their height, so a higher w/h ratio.
# Children are narrower relative to their height → lower ratio.
#
# This is camera-angle-agnostic: it doesn't matter how far
# away the person is or how elevated the camera is, because
# we compare width to height of the *same* bbox.
#
# Typical values observed on CCTV footage:
#   Adult  w/h ≈ 0.35 – 0.55
#   Child  w/h ≈ 0.20 – 0.35
#
# CHILD_ASPECT_RATIO_MAX is the upper bound below which a
# detection is called a child.  Tune for your scene.
# ----------------------------------------------------------

CHILD_ASPECT_RATIO_MAX = 0.15   # w/h  —  below this → child, will have to be adjusted with real camera video

# ==========================================================
# YOLO DETECTOR
# ==========================================================

class YoloDetector:

    def __init__(
        self,
        model_path: str = "models/yolo11x.pt",
        conf_threshold: float = 0.35,
        imgsz: int = 1280,
        frame_skip: int = 2
    ):
        self.tracked_objects     = {}
        self.max_missing_frames  = 10
        self.max_size_multiplier = 1.10
        self.frame_skip          = frame_skip
        self._frame_counter      = -1   # FIX: start at -1 so frame 0 triggers detection
        self._last_boxes         = []

        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.imgsz          = imgsz

        try:
            logging.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)

            self.device = "cpu"

            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.to(self.device)
                logging.info("CUDA acceleration enabled")
            else:
                logging.info("Running on CPU")

            self.use_half    = self.device == "cuda"
            self.initialized = True

            logging.info("YOLO pose model successfully loaded")

        except Exception as e:
            logging.exception(f"Failed to load model: {e}")
            self.initialized = False

    # ======================================================
    # KALMAN FILTER  —  8-state (pos + velocity)
    # ======================================================

    def _create_kalman_filter(self, cx, cy, w, h):

        kf = cv2.KalmanFilter(8, 4)

        kf.transitionMatrix = np.array([
            [1, 0, 0, 0,  1, 0, 0, 0],
            [0, 1, 0, 0,  0, 1, 0, 0],
            [0, 0, 1, 0,  0, 0, 1, 0],
            [0, 0, 0, 1,  0, 0, 0, 1],
            [0, 0, 0, 0,  1, 0, 0, 0],
            [0, 0, 0, 0,  0, 1, 0, 0],
            [0, 0, 0, 0,  0, 0, 1, 0],
            [0, 0, 0, 0,  0, 0, 0, 1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1
        kf.measurementMatrix[1, 1] = 1
        kf.measurementMatrix[2, 2] = 1
        kf.measurementMatrix[3, 3] = 1

        kf.processNoiseCov     = np.eye(8, dtype=np.float32) * 0.1
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        kf.errorCovPost        = np.eye(8, dtype=np.float32)

        kf.statePost = np.array(
            [[cx], [cy], [w], [h], [0.], [0.], [0.], [0.]],
            dtype=np.float32
        )

        return kf

    # ======================================================
    # EXTRACT FACE ROI FROM POSE KEYPOINTS
    # ======================================================

    def _face_roi_from_keypoints(self, keypoints, frame_h, frame_w):

        head_indices = [
            KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE,
            KP_LEFT_EAR, KP_RIGHT_EAR
        ]

        pts = []

        for idx in head_indices:

            if idx >= len(keypoints):
                continue

            kp = keypoints[idx]

            if len(kp) >= 3 and kp[2] < 0.3:
                continue

            x, y = float(kp[0]), float(kp[1])

            if x > 0 and y > 0:
                pts.append((x, y))

        if len(pts) < 2:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        cx     = sum(xs) / len(xs)
        cy     = sum(ys) / len(ys)
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        radius = max(span_x, span_y) * 0.9
        radius = max(radius, 20)

        fx1 = max(0,        int(cx - radius * 1.3))
        fy1 = max(0,        int(cy - radius * 1.6))
        fx2 = min(frame_w,  int(cx + radius * 1.3))
        fy2 = min(frame_h,  int(cy + radius * 1.0))

        return (fx1, fy1, fx2, fy2)

    # ======================================================
    # AGE CLASSIFICATION  —  aspect ratio heuristic
    # ======================================================

    def _classify_age(self, bbox_w: int, bbox_h: int) -> str:
        """
        Classify as 'child' or 'adult' using the bounding box aspect ratio
        (width / height).

        Adults have broader shoulders relative to their height → higher w/h.
        Children are proportionally narrower → lower w/h.

        This approach is robust to camera elevation and viewing distance
        because it compares dimensions *within* the same bbox rather than
        against the frame size.

        Tune CHILD_ASPECT_RATIO_MAX for your specific footage.
        """
        if bbox_h == 0:
            return "adult"

        aspect = bbox_w / bbox_h

        return "child" if aspect < CHILD_ASPECT_RATIO_MAX else "adult"

    # ======================================================
    # DETECTION
    # ======================================================

    def detect(self, frame: np.ndarray):

        if not self.initialized:
            return []

        self._frame_counter += 1

        # --------------------------------------------------
        # FRAME SKIP — predict only, no full inference
        # --------------------------------------------------

        if self._frame_counter % self.frame_skip != 0:

            active_boxes = []

            for data in self.tracked_objects.values():

                prediction = data["kf"].predict()

                cx = float(prediction[0])
                cy = float(prediction[1])
                w  = float(prediction[2])
                h  = float(prediction[3])

                data["bbox"] = (
                    int(cx - w / 2),
                    int(cy - h / 2),
                    int(w),
                    int(h)
                )

                active_boxes.append({
                    "bbox":     data["bbox"],
                    "face_roi": data.get("face_roi"),
                    "age":      data.get("age", "adult"),
                })

            return active_boxes

        # --------------------------------------------------
        # FULL DETECTION FRAME
        # --------------------------------------------------

        original_h, original_w = frame.shape[:2]
        active_boxes = []

        try:

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                half=self.use_half,
                iou=0.4,
                verbose=False
            )

            for data in self.tracked_objects.values():
                data["missing"] += 1

            for result in results:

                if result.boxes is None:
                    continue

                boxes     = result.boxes
                keypoints = result.keypoints

                for i, box in enumerate(boxes):

                    cls_id = int(box.cls[0])

                    if cls_id != 0:
                        continue

                    confidence = float(box.conf[0])

                    if confidence < self.conf_threshold:
                        continue

                    if box.id is None:
                        continue

                    track_id = int(box.id[0])

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    padding_x = int((x2 - x1) * 0.06)
                    padding_y = int((y2 - y1) * 0.06)

                    x1 = max(0, x1 - padding_x)
                    y1 = max(0, y1 - padding_y)
                    x2 = min(original_w, x2 + padding_x)
                    y2 = min(original_h, y2 + padding_y)

                    bw = x2 - x1
                    bh = y2 - y1
                    cx = x1 + bw // 2
                    cy = y1 + bh // 2

                    # Aspect-ratio-based age classification
                    age = self._classify_age(bw, bh)

                    # Face ROI from keypoints
                    face_roi = None

                    if keypoints is not None and i < len(keypoints.data):
                        kps      = keypoints.data[i].cpu().numpy()
                        face_roi = self._face_roi_from_keypoints(
                            kps, original_h, original_w
                        )

                    # Kalman update or init
                    if track_id in self.tracked_objects:

                        prev      = self.tracked_objects[track_id]
                        prev_bbox = prev["bbox"]

                        prev_cx = prev_bbox[0] + prev_bbox[2] // 2
                        prev_cy = prev_bbox[1] + prev_bbox[3] // 2

                        alpha = 0.4

                        smooth_cx = int(prev_cx * alpha + cx * (1 - alpha))
                        smooth_cy = int(prev_cy * alpha + cy * (1 - alpha))
                        smooth_bw = int(prev_bbox[2] * alpha + bw * (1 - alpha))
                        smooth_bh = int(prev_bbox[3] * alpha + bh * (1 - alpha))

                        orig_w, orig_h = prev["original_size"]

                        max_w = int(orig_w * self.max_size_multiplier)
                        max_h = int(orig_h * self.max_size_multiplier)
                        min_w = int(orig_w / self.max_size_multiplier)
                        min_h = int(orig_h / self.max_size_multiplier)

                        smooth_bw = max(min_w, min(smooth_bw, max_w))
                        smooth_bh = max(min_h, min(smooth_bh, max_h))

                        new_orig_w = int(orig_w * 0.95 + bw * 0.05)
                        new_orig_h = int(orig_h * 0.95 + bh * 0.05)

                        measurement = np.array(
                            [[smooth_cx], [smooth_cy], [smooth_bw], [smooth_bh]],
                            dtype=np.float32
                        )

                        prev["kf"].correct(measurement)

                        kf            = prev["kf"]
                        original_size = (new_orig_w, new_orig_h)

                        smooth_box = (
                            smooth_cx - smooth_bw // 2,
                            smooth_cy - smooth_bh // 2,
                            smooth_bw,
                            smooth_bh
                        )

                    else:

                        kf = self._create_kalman_filter(
                            float(cx), float(cy), float(bw), float(bh)
                        )
                        smooth_box    = (x1, y1, bw, bh)
                        original_size = (bw, bh)

                    self.tracked_objects[track_id] = {
                        "bbox":          smooth_box,
                        "face_roi":      face_roi,
                        "age":           age,
                        "missing":       0,
                        "kf":            kf,
                        "original_size": original_size,
                    }

            # Collect active + purge expired
            expired_ids = []

            for track_id, data in self.tracked_objects.items():

                if data["missing"] <= self.max_missing_frames:

                    active_boxes.append({
                        "bbox":     data["bbox"],
                        "face_roi": data.get("face_roi"),
                        "age":      data.get("age", "adult"),
                    })

                else:
                    expired_ids.append(track_id)

            for track_id in expired_ids:
                del self.tracked_objects[track_id]

        except Exception as e:
            logging.exception(f"Detection failed: {e}")

        return active_boxes


# ==========================================================
# DETECTOR SINGLETON
# ==========================================================

DETECTOR = YoloDetector()

# ==========================================================
# SOLID COLOR FILL HELPER
# ==========================================================

def _fill_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 color: tuple) -> np.ndarray:

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return frame

    frame[y1:y2, x1:x2] = color

    return frame

# ==========================================================
# MASKING
# ==========================================================

def _mask_object(
    frame: np.ndarray,
    detection: dict,
    mode: str
) -> np.ndarray:

    bbox     = detection["bbox"]
    face_roi = detection.get("face_roi")
    age      = detection.get("age", "adult")

    color = COLOR_ADULT if age == "adult" else COLOR_CHILD

    x, y, w, h = bbox
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    if mode == "face":

        if face_roi is not None:
            fx1, fy1, fx2, fy2 = face_roi
        else:
            face_cx  = x + w // 2
            face_cy  = y + int(h * 0.15)
            radius_x = int(w * 0.45)
            radius_y = int(h * 0.22)

            fx1 = face_cx - radius_x
            fy1 = face_cy - radius_y
            fx2 = face_cx + radius_x
            fy2 = face_cy + radius_y

        frame = _fill_region(frame, fx1, fy1, fx2, fy2, color)

    elif mode == "body":

        frame = _fill_region(frame, x, y, x + w, y + h, color)

    return frame

# ==========================================================
# MAIN PUBLIC FUNCTION
# ==========================================================

def anonymize_frame(
    frame: np.ndarray,
    mode: str = "face"
) -> np.ndarray:

    detections = DETECTOR.detect(frame)

    if not detections:
        return frame

    for detection in detections:
        frame = _mask_object(frame, detection, mode)

    return frame