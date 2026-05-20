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
# YOLO DETECTOR
# ==========================================================

class YoloDetector:

    def __init__(
        self,
        model_path: str = "models/yolo11x.pt",
        conf_threshold: float = 0.35,       # was 0.20 — reduces false positives
        imgsz: int = 1280
    ):
        self.tracked_objects = {}
        self.max_missing_frames = 10        # was 15 — drop stale ghosts faster

        self.max_prediction_growth = 1.005  # was 1.01
        self.max_size_multiplier = 1.10     # was 1.15
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        try:

            logging.info(f"Loading YOLO model: {model_path}")

            self.model = YOLO(model_path)

            # --------------------------------------------------
            # GPU
            # --------------------------------------------------

            self.device = "cpu"

            if torch.cuda.is_available():

                self.device = "cuda"

                self.model.to(self.device)

                logging.info("CUDA acceleration enabled")

            else:
                logging.info("Running on CPU")

            self.use_half = self.device == "cuda"

            self.initialized = True

            logging.info("YOLO model successfully loaded")

        except Exception as e:

            logging.exception(f"Failed to load model: {e}")

            self.initialized = False

    # ======================================================
    # KALMAN FILTER
    # ======================================================

    def _create_kalman_filter(self, x, y):
        kf = cv2.KalmanFilter(4, 2)

        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1    # was 0.03 — more responsive
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1 # was 0.5 — trust detections more
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        kf.statePost = np.array([
            [x], [y], [0.0], [0.0]
        ], dtype=np.float32)

        return kf

    # ======================================================
    # UPSCALE LOW RESOLUTION INPUT
    # ======================================================

    def _enhance_frame(self, frame):
        return frame  # let YOLO handle resizing natively via imgsz

    # ======================================================
    # DETECTION
    # ======================================================

    def detect(self, frame: np.ndarray):

        if not self.initialized:
            return []

        original_h, original_w = frame.shape[:2]

        active_boxes = []

        try:

            enhanced = self._enhance_frame(frame)

            enhanced_h, enhanced_w = enhanced.shape[:2]

            scale_x = original_w / enhanced_w
            scale_y = original_h / enhanced_h

            results = self.model.track(
                enhanced,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                half=self.use_half,
                iou=0.4,          # was 0.5 — catches overlapping people better
                augment=True,     # test-time augmentation improves recall
                verbose=False
            )

            # --------------------------------------------------
            # PREDICT MISSING TRACKS
            # --------------------------------------------------

            for track_id, data in self.tracked_objects.items():

                data["missing"] += 1

                prediction = data["kf"].predict()

                pred_x = int(prediction[0, 0])
                pred_y = int(prediction[1, 0])

                x, y, w, h = data["bbox"]

                original_w_box = data["original_size"][0]
                original_h_box = data["original_size"][1]

                # --------------------------------------------------
                # VERY LIMITED GROWTH
                # --------------------------------------------------

                grow = self.max_prediction_growth

                new_w = int(w * grow)
                new_h = int(h * grow)

                # --------------------------------------------------
                # HARD SIZE LIMIT
                # --------------------------------------------------

                max_w = int(original_w_box * self.max_size_multiplier)
                max_h = int(original_h_box * self.max_size_multiplier)

                new_w = min(new_w, max_w)
                new_h = min(new_h, max_h)

                # --------------------------------------------------
                # CENTERED PREDICTION
                # --------------------------------------------------

                new_x = pred_x - new_w // 2
                new_y = pred_y - new_h // 2

                data["bbox"] = (
                    new_x,
                    new_y,
                    new_w,
                    new_h
                )

            # --------------------------------------------------
            # PROCESS DETECTIONS
            # --------------------------------------------------

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    cls_id = int(box.cls[0])

                    # Person only
                    if cls_id != 0:
                        continue

                    confidence = float(box.conf[0])

                    if confidence < self.conf_threshold:
                        continue

                    if box.id is None:
                        continue

                    track_id = int(box.id[0])

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Scale back
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    # --------------------------------------------------
                    # SMALLER PADDING
                    # --------------------------------------------------

                    padding_x = int((x2 - x1) * 0.06)
                    padding_y = int((y2 - y1) * 0.06)

                    x1 -= padding_x
                    y1 -= padding_y
                    x2 += padding_x
                    y2 += padding_y

                    width = x2 - x1
                    height = y2 - y1

                    current_box = (x1, y1, width, height)

                    center_x = x1 + width // 2
                    center_y = y1 + height // 2

                    # --------------------------------------------------
                    # SMOOTHING
                    # --------------------------------------------------

                    if track_id in self.tracked_objects:

                        prev = self.tracked_objects[track_id]["bbox"]

                        alpha = 0.4

                        smooth_box = (
                            int(prev[0] * alpha + current_box[0] * (1 - alpha)),
                            int(prev[1] * alpha + current_box[1] * (1 - alpha)),
                            int(prev[2] * alpha + current_box[2] * (1 - alpha)),
                            int(prev[3] * alpha + current_box[3] * (1 - alpha))
                        )

                        measurement = np.array(
                            [[center_x], [center_y]],
                            dtype=np.float32
                        )

                        self.tracked_objects[track_id]["kf"].correct(
                            measurement
                        )

                        kf = self.tracked_objects[track_id]["kf"]

                        # Preserve original stable size
                        original_size = self.tracked_objects[track_id]["original_size"]

                    else:

                        smooth_box = current_box

                        kf = self._create_kalman_filter(
                            np.float32(center_x),
                            np.float32(center_y)
                        )

                        original_size = (width, height)

                    # --------------------------------------------------
                    # SAVE TRACK
                    # --------------------------------------------------

                    self.tracked_objects[track_id] = {
                        "bbox": smooth_box,
                        "missing": 0,
                        "kf": kf,
                        "original_size": original_size
                    }

            # --------------------------------------------------
            # KEEP TRACKS TEMPORARILY
            # --------------------------------------------------

            expired_ids = []

            for track_id, data in self.tracked_objects.items():

                if data["missing"] <= self.max_missing_frames:

                    active_boxes.append(data["bbox"])

                else:

                    expired_ids.append(track_id)

            # Cleanup
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
# MASKING
# ==========================================================

def _mask_object(
    frame: np.ndarray,
    bbox,
    mode: str
):

    x, y, w, h = bbox

    # ------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------

    x = max(0, x)
    y = max(0, y)

    w = max(1, w)
    h = max(1, h)

    # ======================================================
    # FACE MODE
    # ======================================================

    if mode == "face":

        face_x = x + int(w * 0.10)        # was 0.18 — wider crop
        face_y = y - int(h * 0.05)        # start slightly above bbox top

        face_w = int(w * 0.80)            # was 0.64 — covers full head width
        face_h = max(50, int(h * 0.38))   # was 0.28 — covers forehead + chin

        pad_x = int(face_w * 0.10)
        pad_y = int(face_h * 0.15)

        face_x -= pad_x
        face_y -= pad_y

        face_w += pad_x * 2
        face_h += pad_y * 2

        face_x = max(0, face_x)
        face_y = max(0, face_y)

        face_x2 = min(frame.shape[1], face_x + face_w)
        face_y2 = min(frame.shape[0], face_y + face_h)

        frame[face_y:face_y2, face_x:face_x2] = (0, 0, 0)

        return frame

    # ======================================================
    # BODY MODE
    # ======================================================

    elif mode == "body":

        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        frame[y:y2, x:x2] = (0, 0, 0)

        return frame

    return frame

# ==========================================================
# MAIN PUBLIC FUNCTION
# ==========================================================

def anonymize_frame(
    frame: np.ndarray,
    mode: str = "blur"
):

    boxes = DETECTOR.detect(frame)

    if not boxes:
        return frame

    for bbox in boxes:

        frame = _mask_object(
            frame,
            bbox,
            mode
        )

    return frame