"""
road_user_tracker.py
=====================
Detects and tracks vehicles, pedestrians, and cyclists for intersection
safety analysis (separate from the person-anonymization pose pipeline in
anonymizer.py, which only cares about *blurring* people, not classifying
road-user type or computing trajectories).

Reuses the same YOLO weights already loaded for anonymization (yolo11x.pt
is trained on COCO, which includes vehicle classes), so no extra model
download is required.

COCO class IDs used
--------------------
  0  person
  1  bicycle
  2  car
  3  motorcycle
  5  bus
  7  truck

We treat tracked road users in two safety-relevant groups:
  VULNERABLE  -> person, bicycle   (the road users we want to protect)
  VEHICLE     -> car, motorcycle, bus, truck   (the potential hazard)

This grouping is intentional: for intersection safety, a vehicle-vehicle
near-miss and a vehicle-pedestrian near-miss are very different risk
categories. We focus PET analysis on VEHICLE -> VULNERABLE pairs, since
that is what actually predicts injury risk to people outside cars.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================================
# CLASS GROUPS
# ==========================================================

COCO_PERSON     = 0
COCO_BICYCLE    = 1
COCO_CAR        = 2
COCO_MOTORCYCLE = 3
COCO_BUS        = 5
COCO_TRUCK      = 7

VULNERABLE_CLASSES = {COCO_PERSON, COCO_BICYCLE}
VEHICLE_CLASSES    = {COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK}
ROAD_USER_CLASSES  = VULNERABLE_CLASSES | VEHICLE_CLASSES

CLASS_NAMES = {
    COCO_PERSON:     "pedestrian",
    COCO_BICYCLE:    "cyclist",
    COCO_CAR:        "car",
    COCO_MOTORCYCLE: "motorcycle",
    COCO_BUS:        "bus",
    COCO_TRUCK:      "truck",
}


def road_user_group(cls_id: int) -> str:
    if cls_id in VULNERABLE_CLASSES:
        return "vulnerable"
    if cls_id in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


# ==========================================================
# TRACK STATE
# ==========================================================

@dataclass
class TrackPoint:
    frame_idx: int
    timestamp_s: float
    cx_px: float
    cy_px: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels


@dataclass
class Track:
    track_id: int
    cls_id: int
    group: str  # "vulnerable" | "vehicle"
    points: list = field(default_factory=list)  # list[TrackPoint]
    missing: int = 0

    @property
    def label(self) -> str:
        return CLASS_NAMES.get(self.cls_id, "object")


# ==========================================================
# SIMPLE IOU TRACKER
# ==========================================================
# A lighter-weight tracker than full Kalman+pose matching used for persons
# in anonymizer.py — vehicles/cyclists only need centroid trajectories for
# speed/PET, not body silhouettes. IoU matching frame-to-frame is sufficient
# at typical intersection-camera frame rates (>=15 fps).

def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class RoadUserTracker:
    """
    Detects vehicles/pedestrians/cyclists per frame and maintains tracks
    with full trajectory history (for downstream PET + speed analysis).

    Uses YOLO's built-in tracker (ByteTrack via model.track()) for ID
    continuity, same mechanism already used for persons in anonymizer.py.
    """

    def __init__(
        self,
        model_path: str = "models/yolo11x.pt",
        conf_threshold: float = 0.30,
        imgsz: int = 960,
        max_missing_frames: int = 15,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_missing_frames = max_missing_frames

        self.tracks: dict[int, Track] = {}
        self.finished_tracks: dict[int, Track] = {}

        try:
            logging.info(f"[RoadUserTracker] Loading model: {model_path}")
            self.model = YOLO(model_path)

            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.device = device
            self.model.to(self.device)
            self.initialized = True
            logging.info(f"[RoadUserTracker] Model ready on {self.device}")

        except Exception as e:
            logging.exception(f"[RoadUserTracker] Failed to load model: {e}")
            self.initialized = False

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, frame_idx: int, timestamp_s: float) -> list[dict]:
        """
        Run detection+tracking on one frame. Returns a list of currently
        active detections (for optional on-frame visualization), and
        internally appends to each track's trajectory history.
        """
        if not self.initialized:
            return []

        active = []

        try:
            results = self.model.track(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                classes=list(ROAD_USER_CLASSES),
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception as e:
            logging.exception(f"[RoadUserTracker] Tracking inference failed: {e}")
            return []

        seen_ids = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                if box.id is None:
                    continue  # untracked detection, skip (no stable ID to attach trajectory to)

                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                seen_ids.add(track_id)

                if track_id not in self.tracks:
                    self.tracks[track_id] = Track(
                        track_id=track_id,
                        cls_id=cls_id,
                        group=road_user_group(cls_id),
                    )

                track = self.tracks[track_id]
                track.missing = 0
                track.points.append(
                    TrackPoint(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        cx_px=cx,
                        cy_px=cy,
                        bbox=(x1, y1, x2, y2),
                    )
                )

                active.append({
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "label": track.label,
                    "group": track.group,
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                })

        # Age out tracks not seen this frame; move expired ones to
        # finished_tracks so memory doesn't grow unbounded on long videos.
        expired = []
        for tid, track in self.tracks.items():
            if tid not in seen_ids:
                track.missing += 1
                if track.missing > self.max_missing_frames:
                    expired.append(tid)

        for tid in expired:
            self.finished_tracks[tid] = self.tracks.pop(tid)

        return active

    # ------------------------------------------------------------------
    def all_tracks(self) -> dict[int, Track]:
        """Return every track seen so far (active + finished), keyed by track_id."""
        merged = dict(self.finished_tracks)
        merged.update(self.tracks)
        return merged

    # ------------------------------------------------------------------
    def finalize(self) -> dict[int, Track]:
        """Call after the video ends to flush all remaining active tracks."""
        for tid, track in list(self.tracks.items()):
            self.finished_tracks[tid] = track
        self.tracks.clear()
        return self.finished_tracks
