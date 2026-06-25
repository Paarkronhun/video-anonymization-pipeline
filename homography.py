"""
homography.py
=============
Ground-plane calibration: converts pixel coordinates from the camera view
into real-world metric coordinates (meters) on the road plane.

This is required for any physically-meaningful speed or PET (Post-Encroachment
Time) calculation — without it, "distance" is just pixels, which means
nothing physically and is not comparable across cameras/zoom levels.

Usage
-----
    homography = Homography(
        image_points=[(120, 800), (1830, 800), (1600, 300), (350, 300)],
        world_points=[(0, 0), (12, 0), (12, 30), (0, 30)],  # meters
    )

    x_m, y_m = homography.pixel_to_world(960, 540)
    dist_m   = homography.pixel_distance_to_meters((x1, y1), (x2, y2))

Calibration notes
------------------
- Provide exactly 4 (or more) point correspondences between the image and
  a real-world ground plane (in meters), e.g. measured from satellite
  imagery, road markings of known width/spacing, or a tape measure.
- Points should be roughly coplanar (the road surface) — this is a planar
  homography, not a full 3D camera model. That's standard and sufficient
  for intersection PET/speed analysis as long as points are on the ground.
- More than 4 points can be supplied; cv2.findHomography will use a
  least-squares fit (RANSAC) which is more robust to measurement error.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Homography:
    """Wraps a planar homography for pixel <-> world (meter) conversion."""

    def __init__(
        self,
        image_points: Sequence[tuple[float, float]],
        world_points: Sequence[tuple[float, float]],
    ):
        if len(image_points) < 4 or len(world_points) < 4:
            raise ValueError(
                "Homography calibration requires at least 4 point pairs "
                f"(got {len(image_points)} image / {len(world_points)} world)."
            )
        if len(image_points) != len(world_points):
            raise ValueError("image_points and world_points must be the same length.")

        self.image_points = np.array(image_points, dtype=np.float32)
        self.world_points = np.array(world_points, dtype=np.float32)

        method = cv2.RANSAC if len(image_points) > 4 else 0
        H, mask = cv2.findHomography(self.image_points, self.world_points, method=method)

        if H is None:
            raise ValueError(
                "Could not compute homography from the given point pairs — "
                "check that points are not collinear and pairs are correctly ordered."
            )

        self.H = H
        self.H_inv = np.linalg.inv(H)

        # Sanity-check: report reprojection error so silent miscalibration
        # doesn't produce confidently-wrong speeds later.
        self._reprojection_error_m = self._check_reprojection_error()
        if self._reprojection_error_m > 0.5:
            logging.warning(
                "[Homography] Reprojection error is %.2f m — calibration points "
                "may be inaccurate or not coplanar. Speeds/PET may be unreliable.",
                self._reprojection_error_m,
            )
        else:
            logging.info(
                "[Homography] Calibrated OK (mean reprojection error %.3f m).",
                self._reprojection_error_m,
            )

    # ------------------------------------------------------------------
    def _check_reprojection_error(self) -> float:
        errors = []
        for img_pt, world_pt in zip(self.image_points, self.world_points):
            projected = self.pixel_to_world(*img_pt)
            err = np.hypot(projected[0] - world_pt[0], projected[1] - world_pt[1])
            errors.append(err)
        return float(np.mean(errors))

    # ------------------------------------------------------------------
    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        """Convert a single pixel coordinate to world-plane meters."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.H)
        return float(world[0, 0, 0]), float(world[0, 0, 1])

    # ------------------------------------------------------------------
    def pixel_points_to_world(self, points: Sequence[tuple[float, float]]) -> np.ndarray:
        """Batch-convert an array of pixel points to world meters. Shape (N, 2)."""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pts, self.H)
        return world.reshape(-1, 2)

    # ------------------------------------------------------------------
    def pixel_distance_to_meters(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """Real-world distance (meters) between two pixel points."""
        w1 = self.pixel_to_world(*p1)
        w2 = self.pixel_to_world(*p2)
        return float(np.hypot(w1[0] - w2[0], w1[1] - w2[1]))

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "Homography":
        """
        Load calibration from a JSON file of the form:
        {
          "image_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
          "world_points":  [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        }
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data["image_points"], data["world_points"])

    # ------------------------------------------------------------------
    def to_json(self, path: str) -> None:
        data = {
            "image_points": self.image_points.tolist(),
            "world_points": self.world_points.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def identity_homography(scale_m_per_px: float = 1.0) -> Homography:
    """
    Fallback "no calibration available" homography: treats the pixel grid as
    a flat plane scaled uniformly by *scale_m_per_px*. NOT accurate for real
    perspective video (distant objects will appear to move slower than they
    really do) — only use this for quick testing without real calibration
    points. Prefer Homography(...) with real measured points for production.
    """
    image_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    world_points = [
        (0, 0),
        (scale_m_per_px, 0),
        (scale_m_per_px, scale_m_per_px),
        (0, scale_m_per_px),
    ]
    return Homography(image_points, world_points)
