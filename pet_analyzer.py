"""
pet_analyzer.py
================
Post-Encroachment Time (PET) based near-miss / conflict detection for
intersection safety monitoring.

What PET means
---------------
PET is the time gap between when one road user leaves a shared point in
space (the "conflict point") and when the second road user arrives at that
same point. It's the standard surrogate safety measure used in traffic
engineering (Allen et al. 1978; Federal Highway Administration surrogate
safety guidance) precisely because it does not require an actual collision
to happen — a small PET means two paths crossed with very little time to
spare, i.e. a near-miss, even if neither party slowed down or noticed.

  PET = |t_arrival_B - t_arrival_A|   at the point where paths A and B cross

A LOW PET (e.g. < 1.0–1.5 s) with both parties moving indicates a real
near-miss. A high PET means the conflict point was shared safely with a
comfortable margin.

This module:
  1. Converts each track's pixel trajectory to world-plane meters using a
     Homography (see homography.py).
  2. Finds the spatial intersection point(s) of a vehicle's path and a
     vulnerable road user's (pedestrian/cyclist) path.
  3. Computes the time each track was nearest to that crossing point, and
     the resulting PET.
  4. Flags conflicts where PET is below a configurable threshold, weighted
     by closing speed (a fast vehicle + low PET is more severe than a slow
     vehicle + low PET).

Why vehicle <-> vulnerable only (not all pairs)
------------------------------------------------
Two pedestrians crossing paths is not a safety-relevant conflict. Two cars
crossing lanes already has decades of dedicated traffic-conflict research
and different severity assumptions. For an intersection safety monitor
whose stated purpose is protecting people outside vehicles, we deliberately
restrict PET computation to (vehicle, pedestrian/cyclist) pairs, since
that's the safety-critical category — and it keeps the report focused and
the number of pairs evaluated tractable (O(V * P) instead of O(N^2)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import numpy as np

try:
    from .homography import Homography
    from .road_user_tracker import Track, TrackPoint
except ImportError:
    # Allow running this module directly (e.g. `python pet_analyzer.py` from
    # within src/) without the package context.
    from homography import Homography
    from road_user_tracker import Track, TrackPoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==========================================================
# CONFIG DEFAULTS
# ==========================================================

# PET below this is flagged as a near-miss conflict.
# Traffic-safety literature commonly uses 1.0-1.5s as a "critical" PET
# threshold for vehicle-pedestrian conflicts; we default to 1.5s (more
# conservative / catches more borderline cases) and let severity tiers
# (see PETConflict.severity) distinguish how serious each one is.
DEFAULT_PET_THRESHOLD_S = 1.5

# Spatial tolerance (meters) for "the two paths crossed here" — real
# trajectories are noisy and discretized per-frame, so two paths rarely
# intersect at an exact point. This is the radius within which we consider
# the two tracks to have shared the same patch of road.
DEFAULT_CONFLICT_RADIUS_M = 1.5

# Minimum speed (m/s) for a road user to be considered "moving" at the
# conflict point. Near-zero-speed encounters (e.g. a pedestrian standing
# at a corner while cars pass at a safe distance) aren't real conflicts.
MIN_MOVING_SPEED_MPS = 0.3

# Minimum number of trajectory points required before we trust a track's
# path/speed at all (filters out 1-2 frame detection blips).
MIN_TRACK_POINTS = 4


# ==========================================================
# DATA STRUCTURES
# ==========================================================

@dataclass
class WorldTrajectory:
    """A track's trajectory converted to world-plane meters, with derived speed."""
    track_id: int
    label: str
    group: str
    timestamps: np.ndarray   # (N,) seconds
    positions: np.ndarray    # (N, 2) meters
    speeds: np.ndarray       # (N-1,) m/s, speed between consecutive points
    frame_indices: list      # (N,) original frame indices, for report linking


@dataclass
class PETConflict:
    vehicle_id: int
    vehicle_label: str
    vulnerable_id: int
    vulnerable_label: str
    conflict_point_m: tuple        # (x, y) meters, the shared location
    pet_seconds: float
    vehicle_arrival_s: float
    vulnerable_arrival_s: float
    vehicle_speed_mps: float       # vehicle speed near the conflict point
    vulnerable_speed_mps: float
    frame_idx: int                 # representative frame for report/thumbnail
    distance_at_closest_m: float   # closest approach distance (sanity context)

    @property
    def severity(self) -> str:
        """
        Severity tiering based on PET, informed by closing speed.
        These thresholds are a reasonable starting point, not a calibrated
        legal/engineering standard -- tune DEFAULT_PET_THRESHOLD_S and these
        cutoffs against your own site's traffic patterns if you need this
        to match a specific safety program's definitions.
        """
        closing_speed = self.vehicle_speed_mps + self.vulnerable_speed_mps
        if self.pet_seconds < 0.5 and closing_speed > 2.0:
            return "critical"
        if self.pet_seconds < 1.0:
            return "high"
        if self.pet_seconds < DEFAULT_PET_THRESHOLD_S:
            return "moderate"
        return "low"


# ==========================================================
# TRAJECTORY CONSTRUCTION
# ==========================================================

def build_world_trajectory(track: Track, homography: Homography) -> WorldTrajectory | None:
    """Convert a pixel-space Track into a world-space WorldTrajectory with speeds."""
    if len(track.points) < MIN_TRACK_POINTS:
        return None

    pts_px = [(p.cx_px, p.cy_px) for p in track.points]
    positions_m = homography.pixel_points_to_world(pts_px)

    timestamps = np.array([p.timestamp_s for p in track.points], dtype=np.float64)
    frame_indices = [p.frame_idx for p in track.points]

    # Speed between consecutive points: distance(m) / dt(s)
    dt = np.diff(timestamps)
    dt = np.where(dt <= 0, np.nan, dt)  # guard against duplicate timestamps
    diffs_m = np.diff(positions_m, axis=0)
    dist_m = np.hypot(diffs_m[:, 0], diffs_m[:, 1])
    speeds = dist_m / dt
    speeds = np.nan_to_num(speeds, nan=0.0)

    return WorldTrajectory(
        track_id=track.track_id,
        label=track.label,
        group=track.group,
        timestamps=timestamps,
        positions=positions_m,
        speeds=speeds,
        frame_indices=frame_indices,
    )


# ==========================================================
# PATH-CROSSING / PET COMPUTATION
# ==========================================================

def _segment_intersection(p1, p2, p3, p4):
    """
    Return the intersection point of segment p1->p2 and segment p3->p4,
    or None if they don't intersect within both segments' extents.
    All points are (x, y) numpy-compatible tuples in meters.
    """
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4

    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3

    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None  # parallel / degenerate

    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = x1 + t * d1x
        iy = y1 + t * d1y
        return (ix, iy)
    return None


def _time_at_position(traj: WorldTrajectory, point_m: tuple, seg_idx: int, t_frac: float) -> float:
    """Interpolate the timestamp at fractional position t_frac along segment seg_idx."""
    t0 = traj.timestamps[seg_idx]
    t1 = traj.timestamps[seg_idx + 1]
    return float(t0 + t_frac * (t1 - t0))


def find_path_crossings(
    vehicle_traj: WorldTrajectory,
    vulnerable_traj: WorldTrajectory,
) -> list[dict]:
    """
    Find every point where the vehicle's path geometry crosses the
    vulnerable road user's path geometry, treating each trajectory as a
    polyline through consecutive (x, y) positions.

    Returns a list of dicts: {point_m, vehicle_arrival_s, vulnerable_arrival_s,
    vehicle_seg_idx, vulnerable_seg_idx}
    """
    crossings = []

    v_pos = vehicle_traj.positions
    p_pos = vulnerable_traj.positions

    for i in range(len(v_pos) - 1):
        for j in range(len(p_pos) - 1):
            pt = _segment_intersection(v_pos[i], v_pos[i + 1], p_pos[j], p_pos[j + 1])
            if pt is None:
                continue

            # Compute t_frac for each segment to interpolate arrival time
            vx0, vy0 = v_pos[i]; vx1, vy1 = v_pos[i + 1]
            seg_len_v = np.hypot(vx1 - vx0, vy1 - vy0)
            t_frac_v = 0.0 if seg_len_v < 1e-9 else np.hypot(pt[0] - vx0, pt[1] - vy0) / seg_len_v

            px0, py0 = p_pos[j]; px1, py1 = p_pos[j + 1]
            seg_len_p = np.hypot(px1 - px0, py1 - py0)
            t_frac_p = 0.0 if seg_len_p < 1e-9 else np.hypot(pt[0] - px0, pt[1] - py0) / seg_len_p

            vehicle_arrival = _time_at_position(vehicle_traj, pt, i, min(max(t_frac_v, 0.0), 1.0))
            vulnerable_arrival = _time_at_position(vulnerable_traj, pt, j, min(max(t_frac_p, 0.0), 1.0))

            crossings.append({
                "point_m": pt,
                "vehicle_arrival_s": vehicle_arrival,
                "vulnerable_arrival_s": vulnerable_arrival,
                "vehicle_seg_idx": i,
                "vulnerable_seg_idx": j,
            })

    return crossings


def _closest_approach(vehicle_traj: WorldTrajectory, vulnerable_traj: WorldTrajectory) -> tuple[float, float]:
    """
    Fallback for paths that never geometrically cross (common with frame
    discretization / noisy detections) but still pass very close to each
    other. Returns (min_distance_m, timestamp_s_of_closest_approach).

    We compare each vehicle point against the vulnerable trajectory
    interpolated to the same timestamp, restricted to the overlapping
    time window of both tracks.
    """
    t_lo = max(vehicle_traj.timestamps[0], vulnerable_traj.timestamps[0])
    t_hi = min(vehicle_traj.timestamps[-1], vulnerable_traj.timestamps[-1])

    if t_hi <= t_lo:
        return float("inf"), float("nan")

    sample_times = np.linspace(t_lo, t_hi, num=max(20, int((t_hi - t_lo) * 10)))

    v_x = np.interp(sample_times, vehicle_traj.timestamps, vehicle_traj.positions[:, 0])
    v_y = np.interp(sample_times, vehicle_traj.timestamps, vehicle_traj.positions[:, 1])
    p_x = np.interp(sample_times, vulnerable_traj.timestamps, vulnerable_traj.positions[:, 0])
    p_y = np.interp(sample_times, vulnerable_traj.timestamps, vulnerable_traj.positions[:, 1])

    dist = np.hypot(v_x - p_x, v_y - p_y)
    min_idx = int(np.argmin(dist))
    return float(dist[min_idx]), float(sample_times[min_idx])


def _speed_near_time(traj: WorldTrajectory, t: float) -> float:
    """Interpolated speed (m/s) of a trajectory at a given timestamp."""
    if len(traj.speeds) == 0:
        return 0.0
    # speeds[k] applies to the interval [timestamps[k], timestamps[k+1]]
    mid_times = (traj.timestamps[:-1] + traj.timestamps[1:]) / 2.0
    return float(np.interp(t, mid_times, traj.speeds))


def _nearest_frame_idx(traj: WorldTrajectory, t: float) -> int:
    idx = int(np.argmin(np.abs(traj.timestamps - t)))
    return traj.frame_indices[idx]


# ==========================================================
# MAIN ANALYZER
# ==========================================================

class PETAnalyzer:
    """
    Computes PET-based near-miss conflicts between vehicles and vulnerable
    road users (pedestrians, cyclists) over a full set of finished tracks.
    """

    def __init__(
        self,
        homography: Homography,
        pet_threshold_s: float = DEFAULT_PET_THRESHOLD_S,
        conflict_radius_m: float = DEFAULT_CONFLICT_RADIUS_M,
        min_moving_speed_mps: float = MIN_MOVING_SPEED_MPS,
    ):
        self.homography = homography
        self.pet_threshold_s = pet_threshold_s
        self.conflict_radius_m = conflict_radius_m
        self.min_moving_speed_mps = min_moving_speed_mps

    # ------------------------------------------------------------------
    def analyze(self, tracks: dict[int, Track]) -> tuple[list[PETConflict], dict[int, WorldTrajectory]]:
        """
        Run full PET analysis over all tracks.

        Returns
        -------
        (conflicts, trajectories)
          conflicts    : list[PETConflict], sorted by ascending PET (most
                         severe first)
          trajectories : dict[track_id -> WorldTrajectory], useful for the
                         report generator to also show speed stats for
                         every road user, not just ones in a conflict.
        """
        trajectories: dict[int, WorldTrajectory] = {}
        for tid, track in tracks.items():
            traj = build_world_trajectory(track, self.homography)
            if traj is not None:
                trajectories[tid] = traj

        vehicle_ids = [tid for tid, t in trajectories.items() if t.group == "vehicle"]
        vulnerable_ids = [tid for tid, t in trajectories.items() if t.group == "vulnerable"]

        conflicts: list[PETConflict] = []

        for v_id, p_id in product(vehicle_ids, vulnerable_ids):
            vehicle_traj = trajectories[v_id]
            vulnerable_traj = trajectories[p_id]

            conflict = self._evaluate_pair(vehicle_traj, vulnerable_traj)
            if conflict is not None:
                conflicts.append(conflict)

        conflicts.sort(key=lambda c: c.pet_seconds)
        return conflicts, trajectories

    # ------------------------------------------------------------------
    def _evaluate_pair(
        self,
        vehicle_traj: WorldTrajectory,
        vulnerable_traj: WorldTrajectory,
    ) -> PETConflict | None:

        # Only consider pairs whose time windows actually overlap — no
        # point computing PET for two tracks that were never on screen
        # at overlapping times.
        t_lo = max(vehicle_traj.timestamps[0], vulnerable_traj.timestamps[0])
        t_hi = min(vehicle_traj.timestamps[-1], vulnerable_traj.timestamps[-1])
        if t_hi <= t_lo:
            return None

        crossings = find_path_crossings(vehicle_traj, vulnerable_traj)

        best_pet = None
        best_record = None

        if crossings:
            for c in crossings:
                pet = abs(c["vehicle_arrival_s"] - c["vulnerable_arrival_s"])
                if best_pet is None or pet < best_pet:
                    best_pet = pet
                    best_record = c

        if best_record is not None:
            conflict_point = best_record["point_m"]
            vehicle_t = best_record["vehicle_arrival_s"]
            vulnerable_t = best_record["vulnerable_arrival_s"]
            closest_dist = 0.0  # true geometric crossing => 0 m apart spatially
        else:
            # No exact geometric crossing (common due to discretization) —
            # fall back to closest-approach-in-time, but only treat it as a
            # candidate conflict if the closest approach is within our
            # spatial tolerance.
            min_dist, t_at_min = _closest_approach(vehicle_traj, vulnerable_traj)
            if min_dist > self.conflict_radius_m:
                return None

            vx = float(np.interp(t_at_min, vehicle_traj.timestamps, vehicle_traj.positions[:, 0]))
            vy = float(np.interp(t_at_min, vehicle_traj.timestamps, vehicle_traj.positions[:, 1]))
            conflict_point = (vx, vy)
            vehicle_t = t_at_min
            vulnerable_t = t_at_min
            closest_dist = min_dist

        vehicle_speed = _speed_near_time(vehicle_traj, vehicle_t)
        vulnerable_speed = _speed_near_time(vulnerable_traj, vulnerable_t)

        # Skip "conflicts" where one party was essentially stationary far
        # from the shared point in time — not a real near-miss, just two
        # paths that happen to geometrically overlap (e.g. parked car,
        # pedestrian waiting at a corner).
        if vehicle_speed < self.min_moving_speed_mps and vulnerable_speed < self.min_moving_speed_mps:
            return None

        pet = abs(vehicle_t - vulnerable_t)
        if pet > self.pet_threshold_s:
            return None

        representative_t = (vehicle_t + vulnerable_t) / 2.0
        frame_idx = _nearest_frame_idx(vehicle_traj, representative_t)

        return PETConflict(
            vehicle_id=vehicle_traj.track_id,
            vehicle_label=vehicle_traj.label,
            vulnerable_id=vulnerable_traj.track_id,
            vulnerable_label=vulnerable_traj.label,
            conflict_point_m=conflict_point,
            pet_seconds=pet,
            vehicle_arrival_s=vehicle_t,
            vulnerable_arrival_s=vulnerable_t,
            vehicle_speed_mps=vehicle_speed,
            vulnerable_speed_mps=vulnerable_speed,
            frame_idx=frame_idx,
            distance_at_closest_m=closest_dist,
        )
