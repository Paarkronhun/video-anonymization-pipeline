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
KP_LEFT_ELBOW  = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP    = 11
KP_RIGHT_HIP   = 12
KP_LEFT_KNEE   = 13
KP_RIGHT_KNEE  = 14
KP_LEFT_ANKLE  = 15
KP_RIGHT_ANKLE = 16

# ==========================================================
# COLOR CONSTANTS  (BGR)
# ==========================================================

COLOR_ADULT = (0,   0,   0)    # Black  — adult
COLOR_CHILD = (255, 255, 255)  # White  — child

# ---------------------------------------------------------------------------
# FIX 1 – KP confidence threshold lowered so partially visible keypoints
# (e.g. a head cut off at the top of frame) still contribute to the hull.
# ---------------------------------------------------------------------------
KP_CONF_THRESHOLD = 0.15   # was 0.3

# Radius (in pixels) added around each keypoint before computing
# the convex hull.  Accounts for body thickness — torso and limbs
# are wider than a single point.  Scaled by bbox width so it
# adapts to viewing distance automatically.
HULL_RADIUS_RATIO = 0.18   # was 0.12 — wider expansion for partial bodies

# Minimum absolute radius so distant (small) detections still get
# a reasonable hull expansion.
HULL_RADIUS_MIN_PX = 8    # was 6

# CHILD_ASPECT_RATIO_MAX  —  w/h below this → child
CHILD_ASPECT_RATIO_MAX = 0.15

# ---------------------------------------------------------------------------
# FIX 2 – Preprocessing parameters
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT   = 2.0   # contrast enhancement strength
CLAHE_TILE_GRID    = (8, 8)

# ---------------------------------------------------------------------------
# FIX 3 – Multi-scale detection: run inference at two resolutions and merge.
# Catches both large nearby persons AND small distant ones.
# ---------------------------------------------------------------------------
IMGSZ_PRIMARY   = 1280   # main pass  (same as before)
IMGSZ_SECONDARY = 640    # second pass at smaller scale for broader context
NMS_IOU_MERGE   = 0.40   # IoU threshold when merging two-pass boxes

# ==========================================================
# PREPROCESSING HELPERS
# ==========================================================

def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE-based contrast enhancement so that low-light or low-contrast
    persons (common in backgrounds) become easier to detect.

    The enhancement is done only on the luminance channel (YCrCb) so colours
    are preserved and the output stays a normal BGR frame.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def _upscale_small_frame(
    frame: np.ndarray,
    min_dim: int = 720
) -> tuple[np.ndarray, float]:
    """
    If the shorter side of the frame is below *min_dim* pixels, upscale it
    so YOLO has more pixels to work with on small/distant persons.
    Returns (processed_frame, scale_factor).  scale_factor == 1.0 when no
    upscaling is applied.
    """
    h, w = frame.shape[:2]
    short = min(h, w)
    if short >= min_dim:
        return frame, 1.0

    scale = min_dim / short
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


# ==========================================================
# NMS UTILITY  (used when merging multi-scale results)
# ==========================================================

def _nms_boxes(
    boxes_xyxy: list[tuple],
    scores: list[float],
    iou_threshold: float = 0.40
) -> list[int]:
    """
    Simple NMS that returns kept indices.
    boxes_xyxy: list of (x1,y1,x2,y2)
    """
    if not boxes_xyxy:
        return []

    boxes  = np.array(boxes_xyxy, dtype=np.float32)
    scores = np.array(scores,     dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)

        inds  = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# ==========================================================
# YOLO DETECTOR
# ==========================================================

class YoloDetector:

    def __init__(
        self,
        model_path: str = "models/yolo11x.pt",
        # FIX 4 – conf lowered to 0.20 so distant/occluded persons are not
        # silently dropped before tracking even sees them.
        conf_threshold: float = 0.20,   # was 0.35
        imgsz: int = IMGSZ_PRIMARY,
        # FIX 5 – frame_skip set to 1 (every frame is a detection frame).
        # Skipping frames is the single biggest source of missed anonymisation
        # because persons visible on skipped frames only get Kalman-predicted
        # masks that may drift off the actual body.
        frame_skip: int = 1,            # was 2
    ):
        self.tracked_objects     = {}
        self.max_missing_frames  = 10
        self.max_size_multiplier = 1.10
        self.frame_skip          = frame_skip
        self._frame_counter      = -1
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

    def _face_roi_from_keypoints(self, keypoints, frame_h, frame_w, bbox_h=None):
        """
        FIX 6 – When head keypoints are absent (person's head extends above
        the frame, or is occluded), fall back to extrapolating the head
        position from shoulder keypoints.
        """
        head_indices = [
            KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE,
            KP_LEFT_EAR, KP_RIGHT_EAR
        ]

        pts = []

        for idx in head_indices:
            if idx >= len(keypoints):
                continue
            kp = keypoints[idx]
            if len(kp) >= 3 and kp[2] < KP_CONF_THRESHOLD:
                continue
            x, y = float(kp[0]), float(kp[1])
            if x > 0 and y > 0:
                pts.append((x, y))

        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx     = sum(xs) / len(xs)
            cy     = sum(ys) / len(ys)
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            radius = max(span_x, span_y) * 0.9
            radius = max(radius, 20)

            fx1 = max(0,       int(cx - radius * 1.4))
            fy1 = max(0,       int(cy - radius * 1.8))  # extra top margin
            fx2 = min(frame_w, int(cx + radius * 1.4))
            fy2 = min(frame_h, int(cy + radius * 1.0))
            return (fx1, fy1, fx2, fy2)

        # ---------------------------------------------------------------
        # FIX 6b – No visible head keypoints: extrapolate from shoulders
        # ---------------------------------------------------------------
        shoulder_pts = []
        for idx in [KP_LEFT_SHLDR, KP_RIGHT_SHLDR]:
            if idx >= len(keypoints):
                continue
            kp = keypoints[idx]
            if len(kp) >= 3 and kp[2] < KP_CONF_THRESHOLD:
                continue
            x, y = float(kp[0]), float(kp[1])
            if x > 0 and y > 0:
                shoulder_pts.append((x, y))

        if shoulder_pts:
            sx  = sum(p[0] for p in shoulder_pts) / len(shoulder_pts)
            sy  = sum(p[1] for p in shoulder_pts) / len(shoulder_pts)
            # Shoulder width as head-size proxy
            if len(shoulder_pts) == 2:
                shldr_w = abs(shoulder_pts[0][0] - shoulder_pts[1][0])
            else:
                shldr_w = bbox_h * 0.25 if bbox_h else 40
            head_r = max(shldr_w * 0.55, 20)
            # Head is roughly 1.0–1.5× head_r above the shoulders
            head_cy = sy - head_r * 1.2

            fx1 = max(0,       int(sx - head_r * 1.3))
            fy1 = max(0,       int(head_cy - head_r * 1.2))
            fx2 = min(frame_w, int(sx + head_r * 1.3))
            fy2 = min(frame_h, int(head_cy + head_r * 1.0))
            return (fx1, fy1, fx2, fy2)

        return None

    # ======================================================
    # BUILD CONVEX HULL POLYGON FROM ALL BODY KEYPOINTS
    # ======================================================

    def _body_polygon_from_keypoints(
        self,
        keypoints,
        frame_h: int,
        frame_w: int,
        bbox: tuple,          # (x1, y1, x2, y2)
    ):
        """
        FIX 7 – The hull is now always padded to the FULL bounding box
        extent so that a person whose keypoints only cover the lower half
        (head/torso above frame) still gets a complete body mask.

        Steps:
          1. Collect confident keypoints (expanded to discs).
          2. Add the four corners of the detection bbox as anchors.
          3. Compute the convex hull of all those points.
        """
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1

        radius = max(
            int(bbox_w * HULL_RADIUS_RATIO),
            HULL_RADIUS_MIN_PX
        )

        pts = []

        for kp in keypoints:
            if len(kp) >= 3 and kp[2] < KP_CONF_THRESHOLD:
                continue
            x, y = float(kp[0]), float(kp[1])
            if x <= 0 or y <= 0:
                continue
            pts.append((x, y))

        # Ensure we have SOMETHING to build a hull from even if no
        # keypoints are confident enough.
        if len(pts) < 3:
            # Fall back to bbox corners only — hull will be the rectangle.
            pts = [
                (x1, y1), (x2, y1),
                (x2, y2), (x1, y2),
            ]
        else:
            # Always include bbox corners so the mask is never *smaller*
            # than the detected bounding box.
            pts += [
                (x1, y1), (x2, y1),
                (x2, y2), (x1, y2),
            ]

        # Expand each point to a disc (8 samples on circumference)
        expanded = []
        angles   = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        for (x, y) in pts:
            for a in angles:
                ex = np.clip(x + radius * np.cos(a), 0, frame_w - 1)
                ey = np.clip(y + radius * np.sin(a), 0, frame_h - 1)
                expanded.append([ex, ey])

        pts_array = np.array(expanded, dtype=np.float32)
        hull      = cv2.convexHull(pts_array)

        return hull.astype(np.int32)

    # ======================================================
    # AGE CLASSIFICATION  —  aspect ratio heuristic
    # ======================================================

    def _classify_age(self, bbox_w: int, bbox_h: int) -> str:
        if bbox_h == 0:
            return "adult"
        aspect = bbox_w / bbox_h
        return "child" if aspect < CHILD_ASPECT_RATIO_MAX else "adult"

    # ======================================================
    # MULTI-SCALE DETECTION PASS
    # ======================================================

    def _run_inference(self, frame: np.ndarray, imgsz: int):
        """
        Run a single YOLO track/predict pass and return the results list.
        Uses track() so ByteTrack IDs are maintained across calls.
        """
        return self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=imgsz,
            conf=self.conf_threshold,
            half=self.use_half,
            iou=0.4,
            verbose=False,
            classes=[0],   # person only — avoids processing irrelevant classes
        )

    # ======================================================
    # DETECTION
    # ======================================================

    def detect(self, frame: np.ndarray):

        if not self.initialized:
            return []

        self._frame_counter += 1

        # --------------------------------------------------
        # FRAME SKIP  (kept for performance; default = 1
        # meaning every frame is a detection frame)
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
                    "hull":     data.get("hull"),
                    "age":      data.get("age", "adult"),
                })

            return active_boxes

        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        # FIX 8 – CLAHE contrast enhancement before inference.
        # Helps detect persons in shadows, backlit scenes, or dull backgrounds.
        proc_frame = _preprocess_frame(frame)

        # FIX 9 – Upscale very small frames so small persons get more pixels.
        proc_frame, up_scale = _upscale_small_frame(proc_frame)

        original_h, original_w = frame.shape[:2]
        active_boxes = []

        try:

            # ------------------------------------------------------
            # MULTI-SCALE INFERENCE
            # FIX 10 – Run at two resolutions and merge with NMS.
            # Primary pass (large imgsz) = fine detail for large persons.
            # Secondary pass (small imgsz) = global context for crowd/BG.
            # ------------------------------------------------------

            results_primary   = self._run_inference(proc_frame, IMGSZ_PRIMARY)
            results_secondary = self._run_inference(proc_frame, IMGSZ_SECONDARY)

            # Combine into one list, secondary first so primary IDs win NMS.
            all_results = list(results_secondary) + list(results_primary)

            # Track all seen track IDs this frame so we can mark missing
            seen_ids = set()

            for data in self.tracked_objects.values():
                data["missing"] += 1

            # Collect raw detections for NMS-based dedup across passes
            raw_boxes   = []   # (x1,y1,x2,y2) in ORIGINAL frame coords
            raw_scores  = []
            raw_meta    = []   # (track_id, i, result_idx) for keypoint lookup

            for r_idx, result in enumerate(all_results):

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

                    # Scale back to original frame if we upscaled
                    if up_scale != 1.0:
                        x1 /= up_scale; y1 /= up_scale
                        x2 /= up_scale; y2 /= up_scale

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    raw_boxes.append((x1, y1, x2, y2))
                    raw_scores.append(confidence)
                    raw_meta.append((track_id, i, r_idx, result))

            # Deduplicate across the two passes
            kept_indices = _nms_boxes(raw_boxes, raw_scores, NMS_IOU_MERGE)

            for idx in kept_indices:

                x1, y1, x2, y2       = raw_boxes[idx]
                track_id, i, r_idx, result = raw_meta[idx]

                # Re-fetch keypoints from the correct result
                keypoints = result.keypoints

                seen_ids.add(track_id)

                # Reset missing counter for re-detected tracks
                if track_id in self.tracked_objects:
                    self.tracked_objects[track_id]["missing"] = 0

                padding_x = int((x2 - x1) * 0.06)
                padding_y = int((y2 - y1) * 0.06)

                x1 = max(0,          x1 - padding_x)
                y1 = max(0,          y1 - padding_y)
                x2 = min(original_w, x2 + padding_x)
                y2 = min(original_h, y2 + padding_y)

                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw // 2
                cy = y1 + bh // 2

                age = self._classify_age(bw, bh)

                # ------------------------------------------
                # FACE ROI + BODY HULL FROM KEYPOINTS
                # ------------------------------------------

                face_roi = None
                hull     = None

                if keypoints is not None and i < len(keypoints.data):

                    kps = keypoints.data[i].cpu().numpy()

                    face_roi = self._face_roi_from_keypoints(
                        kps, original_h, original_w, bbox_h=bh
                    )

                    hull = self._body_polygon_from_keypoints(
                        kps, original_h, original_w,
                        bbox=(x1, y1, x2, y2),
                    )
                else:
                    # FIX 11 – No keypoints at all: still build a full-bbox
                    # hull so the person is masked.
                    dummy_kps = []
                    hull = self._body_polygon_from_keypoints(
                        dummy_kps, original_h, original_w,
                        bbox=(x1, y1, x2, y2),
                    )

                # ------------------------------------------
                # KALMAN UPDATE OR INIT
                # ------------------------------------------

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
                    "hull":          hull,
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
                        "hull":     data.get("hull"),
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
# DRAWING HELPERS
# ==========================================================

def _fill_polygon(
    frame: np.ndarray,
    hull: np.ndarray,
    color: tuple
) -> np.ndarray:
    """Fill a convex hull polygon with a solid color."""
    cv2.fillPoly(frame, [hull], color)
    return frame


def _fill_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple
) -> np.ndarray:
    """Fallback: fill a rectangle when no hull is available."""
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 > x1 and y2 > y1:
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
    hull     = detection.get("hull")
    age      = detection.get("age", "adult")
    color    = COLOR_ADULT if age == "adult" else COLOR_CHILD

    x, y, w, h = bbox
    x = max(0, x);  y = max(0, y)
    w = max(1, w);  h = max(1, h)

    # ======================================================
    # FACE MODE  —  keypoint-derived ROI (with shoulder
    # extrapolation fallback) or ratio-based last resort.
    # ======================================================

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

        frame = _fill_rect(frame, fx1, fy1, fx2, fy2, color)

    # ======================================================
    # BODY MODE  —  convex hull (now always present because
    # _body_polygon_from_keypoints falls back to the bbox).
    # ======================================================

    elif mode == "body":

        if hull is not None:
            frame = _fill_polygon(frame, hull, color)
        else:
            # Ultimate safety net — should rarely be reached.
            frame = _fill_rect(frame, x, y, x + w, y + h, color)

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
