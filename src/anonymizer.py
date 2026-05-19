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
        model_path: str = "models/yolo11s.pt",
        conf_threshold: float = 0.45
    ):

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        try:

            logging.info(f"Loading YOLO model: {model_path}")

            self.model = YOLO(model_path)

            # GPU acceleration
            if torch.cuda.is_available():
                self.model.to("cuda")
                logging.info("CUDA acceleration enabled")
            else:
                logging.info("Running on CPU")

            self.initialized = True

            logging.info("YOLO model successfully loaded")

        except Exception as e:

            logging.exception(f"Failed to load model: {e}")

            self.initialized = False

    # ======================================================
    # DETECTION
    # ======================================================

    def detect(self, frame: np.ndarray):

        if not self.initialized:
            return []

        boxes = []

        try:

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    cls_id = int(box.cls[0])

                    # COCO class 0 = person
                    if cls_id != 0:
                        continue

                    confidence = float(box.conf[0])

                    if confidence < self.conf_threshold:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    width = x2 - x1
                    height = y2 - y1

                    boxes.append((x1, y1, width, height))

        except Exception as e:
            logging.exception(f"Detection failed: {e}")

        return boxes

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

    # Frame safety
    x = max(0, x)
    y = max(0, y)

    w = max(1, w)
    h = max(1, h)

    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    roi = frame[y:y2, x:x2]

    if roi.size == 0:
        return frame

    # ------------------------------------------------------
    # BLUR MODE
    # ------------------------------------------------------

    if mode == "blur":

        blurred = cv2.GaussianBlur(
            roi,
            (51, 51),
            0
        )

        frame[y:y2, x:x2] = blurred

    # ------------------------------------------------------
    # BLACK BOX MODE
    # ------------------------------------------------------

    elif mode == "full_bbox":

        frame[y:y2, x:x2] = (0, 0, 0)

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